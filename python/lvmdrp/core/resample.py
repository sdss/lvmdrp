import numpy 
from scipy.signal import correlate
from scipy.signal.windows import tukey
from scipy import interpolate


def template_correlate(s1, s2, apod_window=0.5, method="direct"):
    """
    Compute cross-correlation of the s1 and s2 spectra.

    The s1 and s2 spectra have to be on a log-wavelength grid.
    S1 and s2 are first apodized by a Tukey window in order to minimize edge
    and consequent non-periodicity effects and thus decrease
    high-frequency power in the correlation function. To turn off the
    apodization, use alpha=0.

    Parameters
    ----------
    s1 : :class:`~Spectrum1D`
        The frist, e.g. observed spectrum.
    s2 : :class:`~Spectrum1D`
        The second, e.g. template spectrum, which will be correlated with s1
    apod_window: float, callable, or None
        If a callable, will be treated as a window function for apodization of
        the cross-correlation (should behave like a `~scipy.signal.windows`
        window function, with ``sym=True``). If a float, will be treated as the
        ``alpha`` parameter for a Tukey window (`~scipy.signal.windows.tukey`),
        in units of pixels. If None, no apodization will be performed
    method: str
        If you choose "FFT", the correlation will be done through the use
        of convolution and will be calculated faster (for small spectral
        resolutions it is often correct), otherwise the correlation is determined
        directly from sums (the "direct" method in `~scipy.signal.correlate`).

    Returns
    -------
    (`~numpy.array`, `~numpy.array`)
        Arrays with correlation values and lags
    """

    # apodize (might be a no-op if apodization_window is None)
    s1, s2 = apodize_spectrum(s1, s2, apod_window)

    # Normalize template
    normalization = normalize_for_template_matching(s1, s2)

    # Not sure if we need to actually normalize the template. Depending
    # on the specific data uncertainty, the normalization factor
    # may turn out negative. That causes a flip of the correlation function,
    # in which the maximum (correlation peak) is no longer meaningful.
    if normalization < 0.:
        normalization = 1.

    corr = correlate(s1._data, (s2._data * normalization), method=method)

    # Compute lag
    # wave_l is the wavelength array equally spaced in log space.
    wave_l = s1._wave
    delta_log_wave = numpy.log10(wave_l[1]) - numpy.log10(wave_l[0])
    deltas = (numpy.array(range(len(corr))) - len(corr)/2 + 0.5) * delta_log_wave
    lags = numpy.power(10., deltas) - 1.

    return corr, lags


def apodize_spectrum(s1, apod_window):
    '''
    multiply a spectrum with an apodization window. should be performed after resampling,
    before template matching. apod_window can be None, a callable window, or if a scalar it
    is the alpha parameter of a tukey window.
    '''
    if apod_window is None:
        return s1
    else:
        if callable(apod_window):
            window = apod_window
        else:
            def window(wlen):
                return tukey(wlen, alpha=apod_window)
        return s1 * window(s1.shape[0])


def normalize_for_template_matching(s1, s2):
    """
    Calculate a scale factor to be applied to the s2 spectrum so the
    total flux in s1 and s2 will be the same.

    Parameters
    ----------
    s1 : :class:`~Spectrum1D`
        The observed spectrum.
    s2 : :class:`~Spectrum1D`
        The template spectrum, which needs to be normalized in order to be
        compared with the observed spectrum.

    Returns
    -------
    `float`
        A float which will normalize the template spectrum's flux so that it
        can be compared to the observed spectrum.
    """

    num = numpy.nansum((s1._data*s2._data) / (s1._error**2))
    # We need to limit this sum to where observed_spectrum is not NaN as well.
    s2_filtered = ((s2._data / s1._error)**2)
    s2_filtered = s2_filtered[numpy.where(~numpy.isnan(s1._data))]
    denom = numpy.nansum(s2_filtered)

    return num/denom


def get_logw_grid(w=None, wblue=None, wred=None, delta_log_w=None):
    """
    Get a log-spaced wavelength grid.

    If wavelength limits ``wblue``, ``wred`` are not provided, the function will use
    the limits of the w array.

    For the wavelength step, the function uses either the smallest wavelength
    interval found in w, or takes it from the ``delta_log_wavelength`` parameter.

    Parameters
    ----------
    w: numpy.array
        (linear) wavelength grid
    wblue, wred: float
        Wavelength limits to include in the correlation.
    delta_log_w: float
        Log-wavelength step to use to build the log-wavelength
        scale. If None, use limits defined as explained above.

    Returns
    -------
    wout: numpy.array
        log spaced wavelength grid
    """

    w0 = numpy.log10(wblue) if wblue is not None else numpy.log10(w[0])
    w1 = numpy.log10(wred) if wred is not None else numpy.log10(w[-1])

    dw = delta_log_w if delta_log_w is not None else numpy.min(numpy.log10(numpy.gradient(w)))

    return numpy.logspace(w0, w1, num=int((w1 - w0) / dw), endpoint=True, dtype=numpy.float32)



def make_bins(wavs):
    """ Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins. """
    x = (wavs[:-1] + wavs[1:]) / 2
    edges = numpy.concatenate([[2*x[0]-x[1]], x, [2*x[-1] - x[-2]]])
    widths = numpy.concatenate([edges[1:-1] - edges[:-2], wavs[-1] + (wavs[-1] - wavs[-2])/2])
    return edges, widths



def resample_flux_density(xout, x, flux, ivar=None, extrapolate=False):
    """Returns a flux conserving resampling of an input flux density.
    The total integrated flux is conserved.

    Args:
        - xout: output SORTED vector, not necessarily linearly spaced
        - x: input SORTED vector, not necessarily linearly spaced
        - flux: input flux density dflux/dx sampled at x

    both x and xout must represent the same quantity with the same unit

    Options:
        - ivar: weights for flux; default is unweighted resampling
        - extrapolate: extrapolate using edge values of input array, default is False,
          in which case values outside of input array are set to zero.
    
    Setting both ivar and extrapolate raises a ValueError because one cannot
    assign an ivar outside of the input data range. 

    Returns:
        if ivar is None, returns outflux
        if ivar is not None, returns outflux, outivar

    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density
    """
    if ivar is None:
        return _unweighted_resample(xout, x, flux, extrapolate=extrapolate)
    else:
        if extrapolate :
            raise ValueError("Cannot extrapolate ivar. Either set ivar=None and extrapolate=True or the opposite")
        a = _unweighted_resample(xout, x, flux*ivar, extrapolate=False)
        b = _unweighted_resample(xout, x, ivar, extrapolate=False)
        mask = (b>0)
        outflux = numpy.zeros(a.shape)
        outflux[mask] = a[mask] / b[mask]
        dx = numpy.gradient(x)
        dxout = numpy.gradient(xout)
        outivar = _unweighted_resample(xout, x, ivar/dx)*dxout
        
        return outflux, outivar


def resample_flux(xout, x, flux, extrapolate=False):
    """Returns a flux conserving resampling of an input flux.
    The total integrated flux is conserved.

    Args:
        - xout: output SORTED vector, not necessarily linearly spaced
        - x: input SORTED vector, not necessarily linearly spaced
        - flux: input flux sampled at x

    both x and xout must represent the same quantity with the same unit

    Options:
        - extrapolate: extrapolate using edge values of input array, default is False,
          in which case values outside of input array are set to zero.
    
    Returns:
        returns outflux

    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density
    """
    flux_dens = flux/numpy.gradient(x)
    f = _unweighted_resample(xout, x, flux_dens, extrapolate=extrapolate)
    return f*numpy.gradient(xout)


def _unweighted_resample(output_x, input_x, input_flux_density, extrapolate=False) :
    """Returns a flux conserving resampling of an input flux density.
    The total integrated flux is conserved.

    Args:
        output_x: SORTED vector, not necessarily linearly spaced
        input_x: SORTED vector, not necessarily linearly spaced
        input_flux_density: input flux density dflux/dx sampled at x

    both must represent the same quantity with the same unit
    input_flux_density =  dflux/dx sampled at input_x
    
    Options:
        extrapolate: extrapolate using edge values of input array, default is False,
                     in which case values outside of input array are set to zero

    Returns:
        returns output_flux

    This interpolation conserves flux such that, on average,
    output_flux_density = input_flux_density

    The input flux density outside of the range defined by the edges of the first
    and last bins is considered null. The bin size of bin 'i' is given by (x[i+1]-x[i-1])/2
    except for the first and last bin where it is (x[1]-x[0]) and (x[-1]-x[-2])
    so flux density is zero for x<x[0]-(x[1]-x[0])/2 and x>x[-1]+(x[-1]-x[-2])/2

    The input is interpreted as the nodes positions and node values of
    a piece-wise linear function::

        y(x) = sum_i y_i * f_i(x)

    with::
        f_i(x) =    (x_{i-1}<x<=x_{i})*(x-x_{i-1})/(x_{i}-x_{i-1})
                + (x_{i}<x<=x_{i+1})*(x-x_{i+1})/(x_{i}-x_{i+1})

    the output value is the average flux density in a bin
    flux_out(j) = int_{x>(x_{j-1}+x_j)/2}^{x<(x_j+x_{j+1})/2} y(x) dx /  0.5*(x_{j+1}+x_{j-1})

    """
    def interpolate_mask(x, y, mask, kind="linear", fill_value=0):
        if not numpy.any(mask):
            return y
        if numpy.all(mask):
            return y
        known_x, known_v = x[~mask], y[~mask]
        missing_x = x[mask]
        missing_idx = numpy.where(mask)

        f = interpolate.interp1d(known_x, known_v, kind=kind, fill_value=fill_value, bounds_error=False)
        yy = y.copy()
        yy[missing_idx] = f(missing_x)

        return yy

    # shorter names
    ix = input_x
    ox = output_x

    iy = interpolate_mask(input_x, input_flux_density, ~numpy.isfinite(input_flux_density))

    # boundary of output bins
    bins = numpy.zeros(ox.size+1)
    bins[1:-1] = (ox[:-1]+ox[1:])/2.
    bins[0] = 1.5*ox[0]-0.5*ox[1]     #  = ox[0]-(ox[1]-ox[0])/2
    bins[-1] = 1.5*ox[-1]-0.5*ox[-2]  #  = ox[-1]+(ox[-1]-ox[-2])/2
    
    # make a temporary node array including input nodes and output bin bounds
    # first the boundaries of output bins
    tx = bins.copy()

    # if we do not extrapolate,
    # because the input is a considered a piece-wise linear function, i.e. the sum of triangles f_i(x),
    # we add two points at ixmin = ix[0]-(ix[1]-ix[0]) and  ixmax = ix[-1]+(ix[-1]-ix[-2])
    # with zero flux densities, corresponding to the edges of the first and last triangles.
    # this solves naturally the edge problem.
    if not extrapolate :
        # note we have to keep the array sorted here because we are going to use it for interpolation
        ix = numpy.append( 2*ix[0]-ix[1] , ix)
        iy = numpy.append(0.,iy)
        ix = numpy.append(ix, 2*ix[-1]-ix[-2])
        iy = numpy.append(iy, 0.)

    # this sets values left and right of input range to first and/or last input values
    # first and last values are = 0 if we are not extrapolating
    ty = numpy.interp(tx,ix,iy)
    
    #  add input nodes which are inside the node array
    k = numpy.where((ix >= tx[0])&(ix <= tx[-1]))[0]
    if k.size :
        tx = numpy.append(tx,ix[k])
        ty = numpy.append(ty,iy[k])
        
    # sort this node array
    p  =  tx.argsort()
    tx = tx[p]
    ty = ty[p]
    
    # now we do a simple integration in each bin of the piece-wise
    # linear function of the temporary nodes

    # integral of individual trapezes
    trapeze_integrals = (ty[1:]+ty[:-1])*(tx[1:]-tx[:-1])/2.

    # output flux
    # for each bin, we sum the trapeze_integrals that belong to that bin
    # and divide by the bin size
    trapeze_centers = (tx[1:]+tx[:-1])/2.
    binsize  =  bins[1:]-bins[:-1]

    if numpy.any(binsize<=0)  :
        raise ValueError("Zero or negative bin size")
    
    return numpy.histogram(trapeze_centers, bins=bins, weights=trapeze_integrals)[0] / binsize
