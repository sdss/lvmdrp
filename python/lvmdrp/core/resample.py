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
        outivar = _unweighted_resample(xout, x, ivar)

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



def grid_npix(rng=None, dx=None, log=False, base=10.0, default=None):
    """
    Determine the number of pixels needed for a given grid.

    Args:
        rng (array-like, optional):
            Two-element array with the
            starting and ending x coordinate of the pixel centers to
            divide into pixels of a given width.  If *log* is True, this
            must still be the linear value of the x coordinate, not
            log(x)!.
        dx (:obj:`float`, optional):
            Linear or logarithmic pixel width.
        log (:obj:`bool`, optional):
            Flag that the range should be logarithmically binned.
        base (:obj:`float`, optional):
            Base for the logarithm
        default (:obj:`int`, optional):
            Default number of pixels to use. The default is returned
            if either ``rng`` or ``dx`` are not provided.

    Returns:
        :obj:`tuple`: Returns the number of pixels to cover ``rng``
        with pixels of width ``dx`` and a two-element
        `numpy.ndarray`_ with the adjusted range such that number of
        pixels of size dx is the exact integer.

    Raises:
        ValueError:
            Raised if the range is not a two-element vector.
    """
    # If the range or sampling are not provided, the number of pixels is
    # already set
    if rng is None or dx is None:
        return default, rng
    if len(rng) != 2:
        raise ValueError('Range must be a 2-element vector.')

    _rng = numpy.atleast_1d(rng).copy()
    npix = int(numpy.floor(numpy.diff(numpy.log(_rng))[0]/numpy.log(base)/dx) + 1) if log else \
                    int(numpy.floor(numpy.diff(_rng)[0]/dx) + 1)
    _rng[1] = numpy.power(base, numpy.log(_rng[0])/numpy.log(base) + dx*(npix-1)) \
                            if log else _rng[0] + dx*(npix-1)

    # Fix for numerical precision
    if (not log and numpy.isclose(rng[1] - _rng[1], dx)) \
           or (log and numpy.isclose((numpy.log(rng[1]) - numpy.log(_rng[1]))/numpy.log(base), dx)):
        npix += 1
        _rng[1] = numpy.power(base, numpy.log(_rng[0])/numpy.log(base) + dx*(npix-1)) \
                                if log else _rng[0] + dx*(npix-1)

    return npix, _rng


def grid_borders(rng, npix, log=False, base=10.0):
    """
    Determine the borders of bin edges in a grid.

    Args:
        rng (array-like):
            Two-element array with the (geometric) centers of the
            first and last pixel in the grid.
        npix (:obj:`int`):
            Number of pixels in the grid.
        log (:obj:`bool`, optional):
            The input range is (to be) logarithmically sampled.
        base (:obj:`float`, optional):
            The base of the logarithmic sampling. Use
            ``numpy.exp(1.)`` for the natural logarithm.

    Returns:
        :obj:`tuple`: Returns a `numpy.ndarray`_ with the grid
        borders with shape ``(npix+1,)`` and the step size per grid
        point. If ``log=True``, the latter is the geometric step.
    """
    if log:
        _rng = numpy.log(rng)/numpy.log(base)
        dlogx = numpy.diff(_rng)[0]/(npix-1.)
        borders = numpy.power(base, numpy.linspace(*(_rng/dlogx + [-0.5, 0.5]), num=npix+1)*dlogx)
        return borders, dlogx
    dx = numpy.diff(rng)[0]/(npix-1.)
    borders = numpy.linspace(*(numpy.atleast_1d(rng)/dx + numpy.array([-0.5, 0.5])), num=npix+1)*dx
    return borders, dx


def grid_centers(rng, npix, log=False, base=10.0):
    """
    Determine the (geometric) center of pixels in a grid.

    Args:
        rng (array-like):
            Two-element array with the (geometric) centers of the
            first and last pixel in the grid.
        npix (:obj:`int`):
            Number of pixels in the grid.
        log (:obj:`bool`, optional):
            The input range is (to be) logarithmically sampled.
        base (:obj:`float`, optional):
            The base of the logarithmic sampling. Use
            ``numpy.exp(1.)`` for the natural logarithm.

    Returns:
        :obj:`tuple`: Returns a `numpy.ndarray`_ with the grid pixel
        (geometric) ceners with shape ``(npix,)`` and the step size
        per grid point. If ``log=True``, the latter is the geometric
        step.
    """
    if log:
        _rng = numpy.log(rng)/numpy.log(base)
        dlogx = numpy.diff(_rng)[0]/(npix-1.)
        centers = numpy.power(base, numpy.linspace(*(_rng/dlogx), num=npix)*dlogx)
        return centers, dlogx
    dx = numpy.diff(rng)[0]/(npix-1.)
    centers = numpy.linspace(*(numpy.atleast_1d(rng)/dx), num=npix)*dx
    return centers, dx


def borders_to_centers(borders, log=False):
    """
    Convert a set of bin borders to bin centers.

    Grid borders need not be regularly spaced.

    Args:
        borders (`numpy.ndarray`_):
            Borders for adjoining bins.
        log (:obj:`bool`, optional):
            Return the geometric center instead of the linear center
            of the bins.

    Returns:
        `numpy.ndarray`_: The vector of bin centers.
    """
    return numpy.sqrt(borders[:-1]*borders[1:]) if log else (borders[:-1]+borders[1:])/2.0


def centers_to_borders(x, log=False):
    """
    Convert a set of bin centers to bounding edges.

    Grid centers need not be regularly spaced. The first edge of the
    first bin and the last edge of the last bin are assumed to be
    equidistant from the center of the 2nd and penultimate bins,
    respectively.

    Args:
        x (`numpy.ndarray`_):
            Centers of adjoining bins.
        log (:obj:`bool`, optional):
            Adopt a geometric binning instead of a linear binning.

    Returns:
        `numpy.ndarray`_: The vector with the coordinates of
        adjoining bin edges.
    """
    if log:
        dx = numpy.diff(numpy.log(x))
        return numpy.exp(numpy.append(numpy.log(x[:-1]) - dx/2,
                                      numpy.log(x[-1]) + numpy.array([-1,1])*dx[-1]/2))
    dx = numpy.diff(x)
    return numpy.append(x[:-1] - dx/2, x[-1] + numpy.array([-1,1])*dx[-1]/2)


class Resample:
    r"""
    Resample regularly or irregularly sampled data to a new grid using
    integration.
    
    This is a generalization of the routine
    :func:`ppxf.ppxf_util.log_rebin` provided by Michele Cappellari in
    the pPXF package.

    The abscissa coordinates (``x``) or the pixel borders
    (``xBorders``) for the data (``y``) should be provided for
    irregularly sampled data. If the input data is linearly or
    geometrically sampled (``inLog=True``), the abscissa coordinates
    can be generated using the input range for the (geometric) center
    of each grid point. If ``x``, ``xBorders``, and ``xRange`` are
    all None, the function assumes grid coordinates of ``x =
    numpy.arange(y.shape[-1])``.

    The function resamples the data by constructing the borders of
    the output grid using the ``new*`` keywords and integrating the
    input function between those borders. The output data will be set
    to ``ext_value`` for any data beyond the abscissa limits of the
    input data.

    The data to resample (``y``) can be a 1D or 2D array; the
    abscissa coordinates must always be 1D. If ``y`` is 2D, the
    resampling is performed along the last axis (i.e., ``axis=-1``).

    The nominal assumption is that the provided function is a step
    function based on the provided input (i.e., ``step=True``). If
    the output grid is substantially finer than the input grid, the
    assumption of a step function will be very apparent. To assume
    the function is instead linearly interpolated between each
    provided point, choose ``step=False``; higher-order
    interpolations are not provided.

    If errors are provided, a nominal error propagation is performed
    to provide the errors in the resampled data.

    .. warning::

        Depending on the details of the resampling, the output errors
        are likely highly correlated.  Any later analysis of the
        resampled function should account for this.

    The covariance in the resampled pixels can be constructed by
    setting ``covar=True``; however, this is currently only supported
    when ``step=True``. If no errors are provided and ``covar=True``,
    the computed matrix is the *correlation* matrix instead of the
    *covariance* matrix. Given that the resampling is the same for all
    vectors, only one correlation matix will be calculated if no
    errors are provided, even if the input ``y`` is 2D. If the input
    data to be resampled is 2D and errors *are* provided, a
    covariance matrix is calculated for *each* vector in ``y``.
    Beware that this can be an expensive computation.

    The ``conserve`` keyword sets how the units of the input data
    should be treated. If ``conserve=False``, the input data are
    expected to be in density units (i.e., per ``x`` coordinate unit)
    such that the integral over :math:`dx` is independent of the
    units of :math:`x` (i.e., flux per unit angstrom or flux
    density). If ``conserve=True``, the value of the data is assumed
    to have been integrated over the size of each pixel (i.e., units
    of flux). If ``conserve=True``, :math:`y` is converted to units
    of per step in :math:`x` such that the integral before and after
    the resample is the same. For example, if :math:`y` is a spectrum
    in units of flux, the function first converts the units to flux
    density and then computes the integral over each new pixel to
    produce the new spectra with units of flux.

    Args:
        y (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
            Data values to resample. The shape can be 1D or 2D. If
            1D, the shape must be :math:`(N_{\rm pix},)`; otherwise,
            it must be :math:`(N_y,N_{\rm pix})`. I.e., the length of
            the last axis must match the input coordinates.
        e (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Errors in the data that should be resampled. The shape
            must match the input ``y`` array. These data are used to
            perform a nominal calculation of the error in the
            resampled array.
        mask (`numpy.ndarray`_, optional):
            A boolean array indicating values in ``y`` that should be
            ignored during the resampling (values to ignore have
            ``masked=True``, just like in a `numpy.ma.MaskedArray`_).
            The mask used during the resampling is the union of this
            object and the masks of ``y`` and ``e``, if either are
            provided as `numpy.ma.MaskedArray`_ objects.
        x (`numpy.ndarray`_, optional):
            Abscissa coordinates for the data, which do not need to
            be regularly sampled. If the pixel borders are not
            provided, they are assumed to be half-way between
            adjacent pixels, and the first and last borders are
            assumed to be equidistant about the provided value. If
            these coordinates are not provided, they are determined
            by the input borders, the input range, or just assumed to
            be the indices, :math:`0..N_{\rm pix}-1`.
        xRange (array-like, optional):
            A two-element array with the starting and ending value
            for the coordinates of the centers of the first and last
            pixels in ``y``. Default is :math:`[0,N_{\rm pix}-1]`.
        xBorders (`numpy.ndarray`_, optional):
            An array with the borders of each pixel that must have a
            length of :math:`N_{\rm pix}+1`.
        inLog (:obj:`bool`, optional):
            Flag that the input is logarithmically binned, primarily
            meaning that the coordinates are at the geometric center
            of each pixel and the centers are spaced logarithmically.
            If false, the sampling is expected to be linear.
        newx (array-like, optional):
            Abscissa coordinates for the *output* data, which do not
            need to be a regular grid. If this is provided, the pixel
            borders are assumed to be half-way between adjacent
            pixels, and the first and last borders are assumed to be
            equidistant about the provided value. If these
            coordinates are not provided, they are determined by the
            new range, the new number of pixels, and/or the new pixel
            width (and whether or not the new grid should be
            logarithmically binned). If this is provided,
            ``newRange``, ``newpix``, ``newLog``, and ``newdx`` are
            *all* ignored.
        newRange (array-like, optional):
            A two-element array with the (geometric) centers of the
            first and last pixel in the output vector. If not
            provided, assumed to be the same as the input range.
        newBorders (array-like, optional):
            An array with the borders of each pixel in the resampled
            vectors.
        newpix (:obj:`int`, optional): 
            Number of pixels for the output vector.  If not provided,
            assumed to be the same as the input vector.
        newLog (:obj:`bool`, optional):
            The output vector should be logarithmically binned.
        newdx (:obj:`float`, optional):
            The sampling step for the output vector. If
            `newLog=True`, this must be the change in the *logarithm*
            of :math:`x` for the output vector! If not provided, the
            sampling is set by the output range (see ``newRange``
            above) and number of pixels (see ``newpix`` above).
        base (:obj:`float`, optional):
            The base of the logarithm used for both input and output
            sampling, if specified. The default is 10; use
            ``numpy.exp(1)`` for natural logarithm.
        ext_value (:obj:`float`, optional):
            Set extrapolated values to the provided float. If set to
            None, values are just set to the linear extrapolation of
            the data beyond the provided limits; use `ext_value=None`
            with caution!
        conserve (:obj:`bool`, optional):
            Conserve the integral of the input vector.  For example, if
            the input vector is a spectrum in flux units, you should
            conserve the flux in the resampling; if the spectrum is in
            units of flux density, you do not want to conserve the
            integral.
        step (:obj:`bool`, optional):
            Treat the input function as a step function during the
            resampling integration.  If False, use a linear
            interpolation between pixel samples.
    
    Attributes:
        x (`numpy.ndarray`_):
            The coordinates of the function on input.
        xborders (`numpy.ndarray`_):
            The borders of the input pixel samples.
        y (`numpy.ndarray`_):
            The function to resample.
        e (`numpy.ndarray`_):
            The 1-sigma errors in the function to resample.
        m (`numpy.ndarray`_):
            The boolean mask for the input function.
        outx (`numpy.ndarray`_):
            The coordinates of the function on output.
        outborders (`numpy.ndarray`_):
            The borders of the output pixel samples.
        outy (`numpy.ndarray`_):
            The resampled function.
        oute (`numpy.ndarray`_):
            The resampled 1-sigma errors.
        outf (`numpy.ndarray`_):
            The fraction of each output pixel that includes valid data
            from the input function.

    Raises:
        ValueError:
            Raised if more the one of ``x``, ``xRange``, or
            ``xBorders`` are provided, if more the one of ``newx``,
            ``newRange``, or ``newBorders`` are provided, if ``y`` is
            a `numpy.ndarray`_, if ``y`` is not 1D or 2D, if the
            covariance is requested but ``step`` is False, if the
            shapes of the provided errors or mask do not match ``y``,
            if there is insufficient information to construct the
            input or output grid, or if either ``xRange`` or
            ``newRange`` are not two-element arrays.
    """
    def __init__(self, y, e=None, mask=None, x=None, xRange=None, xBorders=None, inLog=False,
                 newx=None, newRange=None, newBorders=None, newpix=None, newLog=True, newdx=None,
                 base=10.0, ext_value=0.0, conserve=False, step=True):

        # Check operation can be performed and is not ill-posed
        if numpy.sum([inp is not None for inp in [x, xRange, xBorders]]) != 1:
            raise ValueError('One and only one of the x, xRange, and xBorders arguments should be '
                             'provided.')
        if numpy.sum([inp is not None for inp in [newx, newRange, newBorders]]) != 1:
            raise ValueError('One and only one of the newx, newRange, and newBorders arguments '
                             'should be provided.')
        if not isinstance(y, numpy.ndarray):
            raise ValueError('Input vector must be a numpy.ndarray!')
        if y.ndim > 2:
            raise ValueError('Input must be a 1D or 2D array!')

        # Setup the data, errors, and mask
        self.y = y.filled(0.0) if isinstance(y, numpy.ma.MaskedArray) else y.copy()
        self.twod = self.y.ndim == 2
        self.e = None if e is None \
                    else e.filled(0.0) if isinstance(e, numpy.ma.MaskedArray) else e.copy()
        self.m = numpy.zeros(self.y.shape, dtype=bool) if mask is None else mask

        # Check the shapes
        if self.e is not None and self.e.shape != self.y.shape:
            raise ValueError('Error array shape mismatched!')
        if self.m.shape != self.y.shape:
            raise ValueError('Mask array shape mismatched!')

        # Get the union of all the relevant masks
        if isinstance(y, numpy.ma.MaskedArray):
            self.m |= y.mask
        if e is not None and isinstance(e, numpy.ma.MaskedArray):
            self.m |= e.mask

        # Get the input coordinates
        nx = self.y.shape[-1] if x is None and xBorders is None else None
        self.x, self.xborders = self._coordinate_grid(x=x, rng=xRange, nx=nx, borders=xBorders,
                                                      log=inLog, base=base)

        # If conserving integral, assume input is integrated over pixel
        # width and convert to a density function (divide by pixel size)
        if conserve:
            self.y /= (numpy.diff(self.xborders)[None,:] if self.twod \
                                else numpy.diff(self.xborders))

        # Get the output coordinates
        nx = self.x.size \
                if newx is None and newBorders is None and newpix is None and newdx is None \
                else newpix
        self.outx, self.outborders = self._coordinate_grid(x=newx, rng=newRange, nx=nx,
                                                           borders=newBorders, dx=newdx,
                                                           log=newLog, base=base)

        # Perform the resampling
        self.outy = self._resample_step(self.y) if step else self._resample_linear(self.y)
    
        # The mask and errors are always interpolated as a step function
        self.oute = None if self.e is None else self._resample_step(self.e, quad=True)
    
        self.outf = self._resample_step(numpy.logical_not(self.m).astype(int)) \
                        / numpy.diff(self.outborders)

        # Do not conserve the integral over the size of the pixel
        if not conserve:
            self.outy /= (numpy.diff(self.outborders)[None,:] if self.twod \
                            else numpy.diff(self.outborders))
            if self.oute is not None:
                self.oute /= (numpy.diff(self.outborders)[None,:] if self.twod \
                                    else numpy.diff(self.outborders))

        # Set values for extrapolated regions
        if ext_value is not None:
            indx = (self.outborders[:-1] < self.xborders[0]) \
                        | (self.outborders[1:] > self.xborders[-1]) 
            if numpy.sum(indx) > 0:
                self.outy[...,indx] = ext_value
                self.outf[...,indx] = 0.
                if self.oute is not None:
                    self.oute[...,indx] = 0.

    @staticmethod
    def _coordinate_grid(x=None, rng=None, nx=None, dx=None, borders=None, log=False, base=10.0):
        """
        Use the provided information to construct the coordinate grid
        and the grid borders.
        """
        if x is not None and borders is not None:
            raise ValueError('Both x and borders provided.  Do not need to call _coordinate_grid, '
                             'but also _coordinate_grid does not check that x and borders are '
                             'consistenet with one another.')
        # if (x is not None or borders is not None) and rng is not None:
        #     warnings.warn('Provided both x or borders and the range.  Ignoring range.')
        if x is None and borders is not None:
            # Use the borders to set the centers
            return borders_to_centers(borders, log=log), borders
        if x is not None and borders is None:
            # Use the centers to set the borders
            return x, centers_to_borders(x, log=log)

        # After this point, both x and borders should be None
        assert x is None and borders is None, 'Coding logic error'

        if rng is None and nx is None:
            raise ValueError('Insufficient input to construct coordinate grid.')

        if rng is None:
            # Just set the result to a uniform pixel grid
            return numpy.arange(nx, dtype=float) + 0.5, numpy.arange(nx+1, dtype=float)

        # After this point, rng cannot be None
        assert rng is not None, 'Coding logic error'

        # if dx is not None and nx is not None:
        #     warnings.warn('Provided rng, dx, and nx, which over-specifies the grid; rng and nx '
        #                   'take precedence.')
        if nx is not None:
            borders = grid_borders(rng, nx, log=log, base=base)[0]
            return borders_to_centers(borders, log=log), borders

        nx, _rng = grid_npix(rng=rng, dx=dx, log=log, base=base)
        borders = grid_borders(_rng, nx, log=log, base=base)[0]
        return borders_to_centers(borders, log=log), borders

    def _resample_linear(self, v, quad=False):
        """Resample the vectors."""

        # Combine the input coordinates and the output borders
        combinedX = numpy.append(self.outborders, self.x)
        srt = numpy.argsort(combinedX)
        combinedX = combinedX[srt]

        # Get the indices where the data should be reduced
        border = numpy.ones(combinedX.size, dtype=bool)
        border[self.outborders.size:] = False
        k = numpy.arange(combinedX.size)[border[srt]]

        # Calculate the integrand
        if self.twod:
            # Linearly interpolate the input function at the output border positions
            interp = interpolate.interp1d(self.x, v, axis=-1, assume_sorted=True,
                                          fill_value='extrapolate')
            combinedY = numpy.append(interp(self.outborders), v, axis=-1)[:,srt]
            integrand = (combinedY[:,1:]+combinedY[:,:-1])*numpy.diff(combinedX)[None,:]/2.0
        else:
            # Linearly interpolate the input function at the output border positions
            interp = interpolate.interp1d(self.x, v, assume_sorted=True,
                                          fill_value='extrapolate')
            combinedY = numpy.append(interp(self.outborders), v)[srt]
            integrand = (combinedY[1:]+combinedY[:-1])*numpy.diff(combinedX)/2.0

        if quad:
            integrand = numpy.square(integrand)

        # Use reduceat to calculate the integral
        out = numpy.add.reduceat(integrand, k[:-1], axis=-1) if k[-1] == combinedX.size-1 \
                        else numpy.add.reduceat(integrand, k, axis=-1)[...,:-1]
    
        return numpy.sqrt(out) if quad else out

    def _resample_step(self, v, quad=False):
        """Resample the vectors."""

        # Convert y to a step function
        #  - repeat each element of the input vector twice
        _v = numpy.repeat(v, 2, axis=1) if self.twod else numpy.repeat(v, 2)
        #  - repeat each element of the border array twice, and remove
        #  the first and last elements
        _x = numpy.repeat(self.xborders, 2)[1:-1]

        # Combine the input coordinates and the output borders into a
        # single vector
        indx = numpy.searchsorted(_x, self.outborders)
        combinedX = numpy.insert(_x, indx, self.outborders)

        # Insert points at the borders of the output function
        v_indx = indx.copy()
        v_indx[indx >= _v.shape[-1]] = -1
        combinedY = numpy.array([ numpy.insert(__v, indx, __v[v_indx]) for __v in _v ]) \
                            if self.twod else numpy.insert(_v, indx, _v[v_indx])

        # Calculate the integrand
        integrand = combinedY[:,1:]*numpy.diff(combinedX)[None,:] if self.twod else \
                        combinedY[1:]*numpy.diff(combinedX)
        if quad:
            integrand = numpy.square(integrand)

        # Get the indices where the data should be reduced
        border = numpy.insert(numpy.zeros(_x.size, dtype=bool), indx,
                              numpy.ones(self.outborders.size, dtype=bool))
        k = numpy.arange(combinedX.size)[border]

        # Use reduceat to calculate the integral
        out = numpy.add.reduceat(integrand, k[:-1], axis=-1) if k[-1] == combinedX.size-1 \
                    else numpy.add.reduceat(integrand, k, axis=-1)[...,:-1]
        return numpy.sqrt(out) if quad else out

    def _resample_step_matrix(self):
        r"""
        Build a matrix such that

        .. math::
            y = \mathbf{A} x

        where :math:`x` is the input vector, :math:`y` is the resampled
        vector, and :math:`\mathbf{A}` is the matrix operations that
        resamples :math:`x`.
        """
        ny = self.outx.size
        nx = self.x.size

        # Repeat each element of the border array twice, and remove the
        # first and last elements
        _p = numpy.repeat(numpy.arange(self.x.size), 2)
        _x = numpy.repeat(self.xborders, 2)[1:-1]

        # Combine the input coordinates and the output borders into a
        # single vector
        indx = numpy.searchsorted(_x, self.outborders)
        combinedX = numpy.insert(_x, indx, self.outborders)

        # Insert points at the borders of the output function
        p_indx = indx.copy()
        p_indx[indx >= _p.shape[-1]] = -1
        combinedP = numpy.insert(_p, indx, _p[p_indx])

        # Get the indices where the data should be reduced
        border = numpy.insert(numpy.zeros(_x.size, dtype=bool), indx,
                              numpy.ones(self.outborders.size, dtype=bool))
        nn = numpy.where(numpy.logical_not(border))[0][::2]
        k = numpy.zeros(len(combinedX), dtype=int)
        k[border] = numpy.arange(numpy.sum(border))
        k[nn-1] = k[nn-2]
        k[nn] = k[nn-1]
        start,end = numpy.where(border)[0][[0,-1]]

        # Calculate the fraction of each pixel into each output pixel
        fraction = numpy.diff(combinedX[start:end+1])
        # Construct the output matrix
        indx = fraction > 0
        A = numpy.zeros((ny, nx), dtype=float)
        A[k[start:end][indx], combinedP[start:end][indx]] = fraction[indx]
        return A
