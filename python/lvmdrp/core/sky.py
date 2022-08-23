# average all sky fibers into one
# take into account the difference in LSF
# two (super?) sky: one per each sky telescope

# * try entry point: joint spectra, fiberflat, linearized, de-trended?
# 
# * try entry point: joint spectra, fiberflat, flux calibration
# 
# 1st approach:
#   - interpolate (naively) the sky in the science region
#   - run skycorr (yield a model?)
#   - subtract the sky from the science fibers
# 
# 2nd approach:
#   - for each science fiber:
#   - look for N closest sky fibers in wavelength and LSF space and combine them into a (super?) sky
#   - fit model
#   - subtract from science
# 
# 3rd approach: handling emission and continuum sky
#   - process/model emission lines component with skycorr
#   - process/model continuum component using physical models (e.g., moon spectra, etc)
#   - combine emission+cont
#   - subtract joint sky model from science fiber

import numpy as np


def build_eigen_sky(sky_fibers):
    """Returns eigen vectors for the given sky fibers
    

    Parameters
    ----------
    sky_fibers:
        1D linear fiber flat sky fibers
    
    Returns
    -------
    eigen_sky:
        the resulting PCA eigen vectors
    """
    eigen_sky = None
    return eigen_sky


def build_sky(sky_fibers, sci_fibers, method="naive", sci_wave=None, sci_lsf=None, sky_wave=None, sky_lsf=None):
    """Build the optimal sky for the given science fiber


    Parameters
    ----------
    sky_fibers:
        1D linear fiber flat sky fibers
    sci_fiber:
        1D linear fiber flat science fiber for which the best sky will be calculated
    method:
        method to use for building the best sky for the given science fiber.
        Possible values:
            - 'naive': simply naive all given sky fibers into one sky (default, best SNR)
            - 'optimal': find in wavelength and LSF space the closest sky fibers and naive them (best spectral quality)
    
    Returns
    -------
    best_sky:
        sky to use for given fiber
    """
    if method == "naive":
        best_sky = np.median(sky_fibers, axis=0)
    elif method == "optimal":
        # handle the case of missing wave_rss and lsf_rss
        # find closest sky fibers in wavelength and LSF space
        best_sky = None
    return best_sky


def fit_continuum(best_sky, sci_fiber):
    """Returns the best continuum model for sky and science
    
    
    Parameters
    ----------
    best_sky:
        sky spectrum
    sci_fiber:
        target science spectrum
    
    Returns
    -------
    cont:
        sky continuum
    
    """
    # this is already in skycorr using a rolling window median (not a physical model)
    # handle the moon:
    #   - we already have moon spectrum
    cont = None
    return cont


def fit_sky(best_sky, fit_flux_cal=True):
    """Runs skycorr (or similar) to fit/adjust a sky model"""
    # preprocessing of best_sky
    # fit model to best_sky
    # skycorr post mortem
    sky_model = None
    return sky_model


def subtract_sky(sky_model, sci_fiber):
    """Subtracts the given sky model to the given science fiber
    
    
    Parameters
    ----------
    sky_model:
        sky to subtract from the given science fiber
    
    """
    clean_fiber = None
    return clean_fiber


def pca_refine(sci_fibers):
    """Runs PCA refinement on all sky subtracted science fibers
    
    NOTE: I thought PCA was going to come as a refinement step
    after the actual sky subtraction. Something along the lines
    of finding residual sky components in the PCA and then removing
    them.

    Parameters
    ----------
    sci_fibers:
        sky subtracted science fibers

    """
    pass