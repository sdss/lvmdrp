import numpy 
from scipy.sparse import csc_array
from scipy.signal import correlate
from scipy.signal.windows import tukey

def _normalize_for_template_matching(s1, s2):
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



def make_bins(wavs):
    """ Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins. """
    edges = numpy.zeros(wavs.shape[0]+1)
    widths = numpy.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]

    return edges, widths


def rebin_spectra(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    see https://arxiv.org/pdf/1705.05165

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = numpy.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = numpy.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = numpy.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= numpy.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = numpy.sqrt(numpy.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= numpy.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes



def template_correlate(observed_spectrum, template_spectrum,
                       apodization_window=0.5, resample=True, method="direct"):
    """
    Compute cross-correlation of the observed and template spectra.


    After re-sampling into log-wavelength, both observed and template
    spectra are apodized by a Tukey window in order to minimize edge
    and consequent non-periodicity effects and thus decrease
    high-frequency power in the correlation function. To turn off the
    apodization, use alpha=0.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum, which will be correlated with
        the observed spectrum.
    apodization_window: float, callable, or None
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
    observed_spectrum, template_spectrum = _apodize(observed_spectrum, template_spectrum, apodization_window)

    # Normalize template
    normalization = _normalize_for_template_matching(observed_spectrum, template_spectrum)

    # Not sure if we need to actually normalize the template. Depending
    # on the specific data uncertainty, the normalization factor
    # may turn out negative. That causes a flip of the correlation function,
    # in which the maximum (correlation peak) is no longer meaningful.
    if normalization < 0.:
        normalization = 1.

    corr = correlate(observed_spectrum._data, (template_spectrum._data * normalization), method=method)

    # Compute lag
    # wave_l is the wavelength array equally spaced in log space.
    wave_l = observed_spectrum._wave
    delta_log_wave = numpy.log10(wave_l[1]) - numpy.log10(wave_l[0])
    deltas = (numpy.array(range(len(corr))) - len(corr)/2 + 0.5) * delta_log_wave
    lags = numpy.power(10., deltas) - 1.

    return corr, lags


def _apodize(spectrum, template, apodization_window):
    # Apodization. Must be performed after resampling.
    if apodization_window is None:
        clean_spectrum = spectrum
        clean_template = template
    else:
        if callable(apodization_window):
            window = apodization_window
        else:
            def window(wlen):
                return tukey(wlen, alpha=apodization_window)
        clean_spectrum = spectrum * window(len(spectrum.spectral_axis))
        clean_template = template * window(len(template.spectral_axis))

    return clean_spectrum, clean_template


def template_logwl_resample(spectrum, template, wblue=None, wred=None,
                            delta_log_wavelength=None,
                            resampler=None):
    """
    Resample a spectrum and template onto a common log-spaced spectral grid.

    If wavelength limits are not provided, the function will use
    the limits of the merged (observed+template) wavelength scale
    for building the log-wavelength scale.

    For the wavelength step, the function uses either the smallest wavelength
    interval found in the *observed* spectrum, or takes it from the
    ``delta_log_wavelength`` parameter.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum.
    wblue, wred: float
        Wavelength limits to include in the correlation.
    delta_log_wavelength: float
        Log-wavelength step to use to build the log-wavelength
        scale. If None, use limits defined as explained above.
    resampler
        A specutils resampler to use to actually do the resampling.  Defaults to
        using a `~specutils.manipulation.LinearInterpolatedResampler`.

    Returns
    -------
    resampled_observed : :class:`~specutils.Spectrum1D`
        The observed spectrum resampled to a common spectral_axis.
    resampled_template: :class:`~specutils.Spectrum1D`
        The template spectrum resampled to a common spectral_axis.
    """

    # Build an equally-spaced log-wavelength array based on
    # the input and template spectrum's limit wavelengths and
    # smallest sampling interval. Consider only the observed spectrum's
    # sampling, since it's the one that counts for the final accuracy
    # of the correlation. Alternatively, use the wred and wblue limits,
    # and delta log wave provided by the user.
    #
    # We work with separate float and units entities instead of Quantity
    # instances, due to the profusion of log10 and power function calls
    # (they only work on floats)
    if wblue:
        w0 = numpy.log10(wblue)
    else:
        ws0 = numpy.log10(spectrum.spectral_axis[0].value)
        wt0 = numpy.log10(template.spectral_axis[0].value)
        w0 = min(ws0, wt0)

    if wred:
        w1 = numpy.log10(wred)
    else:
        ws1 = numpy.log10(spectrum.spectral_axis[-1].value)
        wt1 = numpy.log10(template.spectral_axis[-1].value)
        w1 = max(ws1, wt1)

    if delta_log_wavelength is None:
        ds = numpy.log10(spectrum.spectral_axis.value[1:]) - numpy.log10(spectrum.spectral_axis.value[:-1])
        dw = ds[numpy.argmin(ds)]
    else:
        dw = delta_log_wavelength

    nsamples = int((w1 - w0) / dw)

    log_wave_array = numpy.ones(nsamples) * w0
    for i in range(nsamples):
        log_wave_array[i] += dw * i

    # Build the corresponding wavelength array
    wave_array = numpy.power(10., log_wave_array) * spectrum.spectral_axis.unit

    # Resample spectrum and template into wavelength array so built
    resampled_spectrum = resampler(spectrum, wave_array)
    resampled_template = resampler(template, wave_array)

    # Resampler leaves Nans on flux bins that aren't touched by it.
    # We replace with zeros. This has the net effect of zero-padding
    # the spectrum and/or template so they exactly match each other,
    # wavelengthwise.
    clean_spectrum_flux = numpy.nan_to_num(resampled_spectrum.flux.value) * resampled_spectrum.flux.unit
    clean_template_flux = numpy.nan_to_num(resampled_template.flux.value) * resampled_template.flux.unit

    clean_spectrum = Spectrum1D(spectral_axis=resampled_spectrum.spectral_axis,
                        flux=clean_spectrum_flux,
                        uncertainty=resampled_spectrum.uncertainty,
                        velocity_convention='optical',
                        rest_value=spectrum.rest_value)
    clean_template = Spectrum1D(spectral_axis=resampled_template.spectral_axis,
                        flux=clean_template_flux,
                        uncertainty=resampled_template.uncertainty,
                        velocity_convention='optical',
                        rest_value=template.rest_value)

    return clean_spectrum, clean_template

def resample_flux(xout, x, flux, ivar=None, extrapolate=False):
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

    The input flux density outside of the range defined by the edges of the first
    and last bins is considered null. The bin size of bin 'i' is given by (x[i+1]-x[i-1])/2
    except for the first and last bin where it is (x[1]-x[0]) and (x[-1]-x[-2])
    so flux density is zero for x<x[0]-(x[1]-x[0])/2 and x>x[-1]-(x[-1]-x[-2])/2

    The input is interpreted as the nodes positions and node values of
    a piece-wise linear function::

        y(x) = sum_i y_i * f_i(x)

    with::

        f_i(x) =    (x_{i-1}<x<=x_{i})*(x-x_{i-1})/(x_{i}-x_{i-1})
                  + (x_{i}<x<=x_{i+1})*(x-x_{i+1})/(x_{i}-x_{i+1})

    the output value is the average flux density in a bin::

        flux_out(j) = int_{x>(x_{j-1}+x_j)/2}^{x<(x_j+x_{j+1})/2} y(x) dx /  0.5*(x_{j+1}+x_{j-1})

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

def _unweighted_resample(output_x,input_x,input_flux_density, extrapolate=False) :
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

    # shorter names
    ix=input_x
    iy=input_flux_density
    ox=output_x

    # boundary of output bins
    bins=numpy.zeros(ox.size+1)
    bins[1:-1]=(ox[:-1]+ox[1:])/2.
    bins[0]=1.5*ox[0]-0.5*ox[1]     # = ox[0]-(ox[1]-ox[0])/2
    bins[-1]=1.5*ox[-1]-0.5*ox[-2]  # = ox[-1]+(ox[-1]-ox[-2])/2
    
    # make a temporary node array including input nodes and output bin bounds
    # first the boundaries of output bins
    tx=bins.copy()

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
    # first and last values are=0 if we are not extrapolating
    ty=numpy.interp(tx,ix,iy)
    
    #  add input nodes which are inside the node array
    k=numpy.where((ix>=tx[0])&(ix<=tx[-1]))[0]
    if k.size :
        tx=numpy.append(tx,ix[k])
        ty=numpy.append(ty,iy[k])
        
    # sort this node array
    p = tx.argsort()
    tx=tx[p]
    ty=ty[p]
    
    # now we do a simple integration in each bin of the piece-wise
    # linear function of the temporary nodes

    # integral of individual trapezes
    trapeze_integrals=(ty[1:]+ty[:-1])*(tx[1:]-tx[:-1])/2.
    
    # output flux
    # for each bin, we sum the trapeze_integrals that belong to that bin
    # and divide by the bin size

    trapeze_centers=(tx[1:]+tx[:-1])/2.
    binsize = bins[1:]-bins[:-1]

    if numpy.any(binsize<=0)  :
        raise ValueError("Zero or negative bin size")
    
    return numpy.histogram(trapeze_centers, bins=bins, weights=trapeze_integrals)[0] / binsize


def project(x1,x2):
    """
    return a (sparse) projection matrix so that arrays are related by linear interpolation
    x1: Array with one binning, must be sorted in ascending order
    x2: new binning, must be sorted in ascending order

    Return Pr: x1= Pr.dot(x2) in the overlap region
    """
    #Pr=numpy.zeros((len(x2),len(x1)))

    e1 = numpy.zeros(len(x1)+1)
    e1[1:-1]=(x1[:-1]+x1[1:])/2.0  # calculate bin edges
    e1[0]=1.5*x1[0]-0.5*x1[1]
    e1[-1]=1.5*x1[-1]-0.5*x1[-2]
    e1lo = e1[:-1]  # make upper and lower bounds arrays vs. index
    e1hi = e1[1:]

    e2=numpy.zeros(len(x2)+1)
    e2[1:-1]=(x2[:-1]+x2[1:])/2.0  # bin edges for resampled grid
    e2[0]=1.5*x2[0]-0.5*x2[1]
    e2[-1]=1.5*x2[-1]-0.5*x2[-2]

    R = []
    C = []
    V = []    
    for ii in range(len(e2)-1): # columns
        #- Find indices in x1, containing the element in x2
        #- This is much faster than looping over rows

        k = numpy.where((e1lo<=e2[ii]) & (e1hi>e2[ii]))[0]
        # this where obtains single e1 edge just below start of e2 bin
        emin = e2[ii]
        emax = e1hi[k]
        if e2[ii+1] < emax: 
            emax = e2[ii+1]
        dx = (emax-emin)/(e1hi[k]-e1lo[k])
        if k.size > 0:
            R.append(ii)
            C.append(k[0])
            V.append(dx[0])
        #Pr[ii,k] = dx    # enter first e1 contribution to e2[ii]

        if e2[ii+1] > emax :
            # cross over to another e1 bin contributing to this e2 bin
            m = numpy.where((e1 < e2[ii+1]) & (e1 > e1hi[k]))[0]
            if len(m) > 0 :
               # several-to-one resample.  Just consider 3 bins max. case
               R.append(ii)
               C.append(k[0]+1)
               V.append(1.0)
               #Pr[ii,k[0]+1] = 1.0  # middle bin fully contained in e2
               q = k[0]+2
            else: 
                q = k[0]+1  # point to bin partially contained in current e2 bin

            try:
                emin = e1lo[q]
                emax = e2[ii+1]
                dx = (emax-emin)/(e1hi[q]-e1lo[q])
                R.append(ii)
                C.append(q)
                V.append(dx)
                # Pr[ii,q] = dx
            except Exception:
                pass

    #- edge:
    if x2[-1]==x1[-1]:
        R.append(len(x2)-1)
        C.append(len(x1)-1)
        V.append(1.0)
        #Pr[-1,-1]=1
    return csc_array((V, (R, C)), shape=(len(x2), len(x1)), dtype=numpy.float32)

def resample_project(outwave, wave, flux,ivar=None):
    """
    rebinning conserving S/N
    Algorithm is based on http://www.ast.cam.ac.uk/%7Erfc/vpfit10.2.pdf
    Appendix: B.1

    Args:
    outwave: new wavelength array
    wave : original wavelength array
    flux : df/dx (Flux per A) sampled at x
    ivar : ivar in original binning. If not None, ivar in new binning is returned.

    Note:
    Full resolution computation for resampling is expensive for quicklook.

    desispec.interpolation.resample_flux using weights by ivar does not conserve total S/N.
    Tests with arc lines show much narrow spectral profile, thus not giving realistic psf resolutions
    This algorithm gives the same resolution as obtained for native CCD binning, i.e, resampling has
    insignificant effect. Details,plots in the arc processing note.
    """
    #- convert flux to per bin before projecting to new bins
    flux=flux*numpy.gradient(wave)

    Pr=project(wave,outwave)
    newflux=Pr.dot(flux)
    #- convert back to df/dx (per angstrom) sampled at outwave
    newflux/=numpy.gradient(outwave) #- per angstrom
    if ivar is None:
        return newflux
    else:
        ivar = ivar/(numpy.gradient(wave))**2.0
        newvar=Pr.dot(ivar**(-1.0)) #- maintaining Total S/N
        # RK:  this is just a kludge until we more robustly ensure newvar is correct
        k = numpy.where(newvar <= 0.0)[0]
        newvar[k] = 0.0000001  # flag bins with no contribution from input grid
        newivar=1/newvar
        # newivar[k] = 0.0

        #- convert to per angstrom
        newivar*=(numpy.gradient(outwave))**2.0
        return newflux, newivar

