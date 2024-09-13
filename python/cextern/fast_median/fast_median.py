import ctypes
import numpy
from numpy.ctypeslib import ndpointer
from scipy.ndimage import median_filter

HAVE_MEDIAN_SO = False
try:
    import platform
    from os import path, environ
    s = platform.system()
    resources_dir = environ.get('LVMDRP_LIB_DIR') or path.join(path.dirname(__file__), 'src')
    if s=='Linux':
        resources_dir = path.join(resources_dir, 'fast_median.so')
        #print(resources_dir)
        lib = ctypes.cdll.LoadLibrary(resources_dir)
    elif s=='Darwin':
        resources_dir = path.join(resources_dir, 'fast_median.dylib')
        #print(resources_dir)
        lib = ctypes.cdll.LoadLibrary(resources_dir)
    else:
        raise Exception('Unknown platform: '+s)

    #template <typename T>
    #void median_filter_2d(int x, int y, int hx, int hy, int blockhint, const T* in, T* out);
    lib.median_filter_2d_float.argtypes = (ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int,
                                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"))
    lib.median_filter_2d_float.restype = None
    lib.median_filter_2d_double.argtypes = (ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int,
                                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"))
    lib.median_filter_2d_double.restype = None

    lib.median_filter_1d_float.argtypes = (ctypes.c_int, 
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"))
    lib.median_filter_1d_float.restype = None
    lib.median_filter_1d_double.argtypes = (ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"))
    lib.median_filter_1d_double.restype = None

    HAVE_MEDIAN_SO = True
except Exception:
    print(Exception)
    pass

print('HAVE_MEDIAN_SO ',HAVE_MEDIAN_SO)

if HAVE_MEDIAN_SO is True:
    def fast_median_filter_2d(input, size=None):
        assert len(input.shape) == 2, 'fast_median_filter_2d requires 2-dimensional input'
        assert len(size) == 2, 'fast_median_filter_2d requires two element size tuple, list or array'
        out = numpy.empty(input.shape, dtype=input.dtype)
        # Note we have to flip the shape and box x/y to match python's indexing
        if input.dtype==numpy.float32:
            lib.median_filter_2d_float(input.shape[1], input.shape[0], size[1], size[0], 0, input, out)
        elif input.dtype==numpy.float64:
            lib.median_filter_2d_double(input.shape[1], input.shape[0], size[1], size[0], 0, input, out)
        else:
            raise TypeError('median_filter_2d not implemented for type '+str(input.dtype))
        return out

    def fast_median_filter_1d(input, size=None):
        assert len(input.shape) == 1, 'fast_median_filter_1d requires 1-dimensional input'
        out = numpy.empty(input.shape, dtype=input.dtype)
        if input.dtype==numpy.float32:
            lib.median_filter_1d_float(input.shape[0], size, 0, input, out)
        elif input.dtype==numpy.float64:
            lib.median_filter_1d_double(input.shape[0], size, 0, input, out)
        else:
            raise TypeError('median_filter_1d not implemented for type '+str(input.dtype))
        return out
else:
    def fast_median_filter_2d(input, size=None):
        return median_filter(input, size=size)
    def fast_median_filter_1d(input, size=None):
        return median_filter(input, size=size)
