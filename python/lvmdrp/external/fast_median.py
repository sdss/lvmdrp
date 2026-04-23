import numpy
from lvmdrp import log


try:
    import fast_median

    def fast_median_filter_2d(input, size=None):
        assert len(input.shape) == 2, 'fast_median_filter_2d requires 2-dimensional input'
        assert len(size) == 2, 'fast_median_filter_2d requires two element size tuple, list or array'
        out = numpy.empty(input.shape, dtype=input.dtype)
        # Note we have to flip the shape and box x/y to match python's indexing
        if input.dtype==numpy.float32:
            fast_median.median_filter_2d_float(input.shape[1], input.shape[0], size[1], size[0], 0, input, out)
        elif input.dtype==numpy.float64:
            fast_median.median_filter_2d_double(input.shape[1], input.shape[0], size[1], size[0], 0, input, out)
        else:
            raise NotImplementedError('median_filter_2d not implemented for type '+str(input.dtype))
        return out

    def fast_median_filter_1d(input, size=None):
        assert len(input.shape) == 1, 'fast_median_filter_1d requires 1-dimensional input'
        out = numpy.empty(input.shape, dtype=input.dtype)
        if input.dtype==numpy.float32:
            fast_median.median_filter_1d_float(input.shape[0], size, 0, input, out)
        elif input.dtype==numpy.float64:
            fast_median.median_filter_1d_double(input.shape[0], size, 0, input, out)
        else:
            raise NotImplementedError('median_filter_1d not implemented for type '+str(input.dtype))
        return out
except ImportError as e:
    log.debug(f"fast_median import failed: {e}")
    log.warning("could not import C++ extension for fast median filtering. Falling back to Scipy's implementation")

    from scipy.ndimage import median_filter

    def fast_median_filter_2d(input, size=None):
        return median_filter(input, size=size, mode="nearest")
    def fast_median_filter_1d(input, size=None):
        return median_filter(input, size=size, mode="nearest")
