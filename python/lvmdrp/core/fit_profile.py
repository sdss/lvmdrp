# -*- coding: utf-8 -*-
from copy import deepcopy as copy
import warnings
import inspect
import itertools
import numpy
import bottleneck as bn
from functools import wraps
from functools import lru_cache

import astropy.io.fits as pyfits
from astropy.modeling.functional_models import Voigt1D, Lorentz1D, Moffat1D
from scipy import interpolate, integrate, optimize, special
from scipy.signal import fftconvolve, convolve

from lvmdrp.core.plot import plt, create_subplots, make_axes_locatable, plot_gradient_fit, plot_radial_gradient_fit


fact = numpy.sqrt(2.0 * numpy.pi)
skew_factor = numpy.sqrt(2 / numpy.pi)

def polyfit2d(x, y, z, order=3):
    """
    Fit 2D polynomial
    """
    ncols = (order + 1) ** 2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x ** i * y ** j
    m, null, null, null = numpy.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m):
    """
    Generate 2D polynomial
    """
    order = int(numpy.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = numpy.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z

def gaussians(pars, x, alpha=2.3, collapse=True):
    """Gaussian models for multiple components"""
    y = pars[0][:, None] * numpy.exp(-0.5 * numpy.abs((x[None, :] - pars[1][:, None]) / pars[2][:, None]) ** alpha) / (pars[2][:, None] * fact)
    if not collapse:
        return y
    return bn.nansum(y, axis=0)

def guess_gaussians_integral(pixels, data, centroids, fwhms, nsigma=6, return_pixels_selection=False):
    fact = numpy.sqrt(2 * numpy.pi)
    integrals = numpy.zeros(len(centroids), dtype=numpy.float32)
    sigmas = fwhms / 2.354

    select = numpy.zeros(pixels.size, dtype="bool")
    for i in range(len(centroids)):
        select_ = numpy.logical_and(
            pixels > centroids[i] - nsigma * sigmas[i],
            pixels < centroids[i] + nsigma * sigmas[i],
        )
        integrals[i] = numpy.interp(centroids[i], pixels, data) * fact * sigmas[i]
        select = numpy.logical_or(select, select_)
    if return_pixels_selection:
        return integrals, select
    return integrals

def moffats(pars, x):
    counts, centroids, fwhms, betas = pars
    r_d = fwhms[:,None] / (2.0 * numpy.sqrt(2 ** (1.0 / betas[:,None]) - 1.0))
    sigma_0 = (betas[:,None] - 1.0) * counts[:,None] / (numpy.pi * (r_d**2))
    return bn.nansum(sigma_0 * (1.0 + ((x[None,:] - centroids[:,None]) / r_d) ** 2) ** (-betas[:,None]), axis=0)

def mexhat(radius, x, normalize_area=True):
    def _model(radius, x):
        return 2 * numpy.abs(numpy.sqrt(radius**2 - x**2))
    def _integral(radius, x):
        return 0.5 * (x * _model(radius, x) + radius**2*numpy.arcsin(x/radius))

    model = _model(radius, x)
    model = numpy.nan_to_num(model, nan=0.0)
    if normalize_area:
        return model / (2*_integral(radius, radius))
    return model

def fiber_profile(centroids, radii, x):
    '''
    semicircular profile: tophat over the unit circle collapsed along one dimension
    '''
    c = numpy.atleast_2d(centroids).T
    r = numpy.atleast_2d(radii).T
    x_ = numpy.atleast_2d(x)
    models = 2 * numpy.sqrt(r**2 - (x_ - c)**2)
    models = numpy.nan_to_num(models, nan=0.0)

    integrals = r**2*numpy.arcsin(1.0)
    models = models / (2*integrals)
    return models

def oversample(x, oversampling_factor):
    x = numpy.asarray(x)
    is_1d = x.ndim == 1

    if is_1d:
        x = x[:, None]  # shape: (nsamples, 1)

    nsamples, nfuncs = x.shape
    dx = numpy.gradient(x, axis=0) / 2.0  # shape: (nsamples, nfuncs)

    # Oversampling offsets
    sub_idx = numpy.arange(oversampling_factor)  # shape: (os,)
    offsets = (sub_idx + 0.5) / oversampling_factor - 0.5  # centered in each subinterval

    # Broadcast for oversampled grid
    x = x[:, :, None]  # shape: (nsamples, nfuncs, 1)
    dx = dx[:, :, None]  # same shape
    oversampled = x + offsets * dx * 2  # shape: (nsamples, nfuncs, os)

    # Reshape: stack oversampling axis next to sample axis
    oversampled = oversampled.transpose(0, 2, 1).reshape(-1, nfuncs)  # (nsamples * os, nfuncs)

    if is_1d:
        return oversampled.ravel()
    return oversampled

def pixelate(x, models, oversampling_factor):
    models_bins = models.reshape((models.shape[0], models.shape[1]//oversampling_factor, oversampling_factor))
    models_pixelated = integrate.trapezoid(models_bins, dx=x[1]-x[0], axis=2)
    return models_pixelated

def update_params(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._pars = self.pack_params(args[0])
        return func(self, *args, **kwargs)
    return wrapper


class IFUGradient(object):

    @classmethod
    def ifu_factors(cls, factors, fiber_groups, normalize=True):
        iid, fid = min(fiber_groups), max(fiber_groups)
        ifu = numpy.ones_like(fiber_groups, dtype="float")
        for spid in range(iid, fid+1):
            ifu[fiber_groups == spid] *= factors[spid-1]
        return ifu / bn.nanmean(ifu) if normalize else ifu

    @classmethod
    def ifu_gradient(cls, coeffs, x, y, normalize=True):
        ncoeffs = len(coeffs)
        order = int(numpy.sqrt(ncoeffs))

        G = numpy.zeros((x.size, ncoeffs))
        ij = itertools.product(range(order), repeat=2)
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        ifu = bn.nansum(G * numpy.asarray(coeffs)[None, :], axis=1)
        return ifu / bn.nanmean(ifu) if normalize else ifu

    @classmethod
    def ifu_joint_model(cls, coeffs, factors, x, y, fiber_groups, normalize=True, return_components=False):
        gradient_model = cls.ifu_gradient(coeffs=coeffs, x=x, y=y, normalize=normalize)
        factors_model = cls.ifu_factors(factors=factors, fiber_groups=fiber_groups, normalize=normalize)
        model = gradient_model * factors_model
        if normalize:
            model /= bn.nanmean(model)
        if return_components:
            return model, gradient_model, factors_model
        return model

    def _get_fixed_selection(self, coeffs_idx, factors_idx):
        fixed_coeffs = numpy.zeros_like(self._guess_coeffs, dtype="bool")
        fixed_factors = numpy.zeros_like(self._guess_factors, dtype="bool")
        fixed_coeffs[coeffs_idx] = True
        fixed_factors[factors_idx] = True
        return fixed_coeffs, fixed_factors

    def _pack_pars(self, coeffs=None, factors=None):
        _coeffs = numpy.where(self._fixed_coeffs, self._guess_coeffs, copy(coeffs or self._coeffs))
        _factors = numpy.where(self._fixed_factors, self._guess_factors, copy(factors or self._factors))
        return _coeffs.tolist(), _factors.tolist()

    def _unpack_pars(self, pars):
        _coeffs = numpy.where(self._fixed_coeffs, self._guess_coeffs, pars[:self._ncoeffs])
        _factors = numpy.where(self._fixed_factors, self._guess_factors, pars[self._ncoeffs:])
        return _coeffs, _factors

    def __init__(self, guess_coeffs, guess_factors, fixed_coeffs=None, fixed_factors=None):
        self._guess_coeffs = list(guess_coeffs)
        self._guess_factors = list(guess_factors)
        self._ncoeffs = len(self._guess_coeffs)
        self._nfactors = len(self._guess_factors)

        # this attributes should remain constant for each instance
        self._fixed_coeffs, self._fixed_factors = self._get_fixed_selection(coeffs_idx=fixed_coeffs or [], factors_idx=fixed_factors or [])

        # initialize parameters
        self._coeffs, self._factors = copy(self._guess_coeffs), copy(self._guess_factors)

    def __call__(self, x, y, fiber_groups, coeffs=None, factors=None):
        _coeffs, _factors = self._pack_pars(coeffs, factors)
        return self.ifu_joint_model(_coeffs, _factors, x, y, fiber_groups)

    def residuals(self, pars, x, y, z, fiber_groups, sigma=None):
        self._coeffs, self._factors = self._unpack_pars(pars)
        model = self(x, y, fiber_groups)
        return (model - z) / (1.0 if sigma is None else sigma)

    def fit(self, x, y, z, fiber_groups, sigma=None):
        guess = self._guess_coeffs + self._guess_factors
        bound_lower = len(self._guess_coeffs) * [-numpy.inf] + len(self._guess_factors) * [0.1]
        bound_upper = len(self._guess_coeffs) * [+numpy.inf] + len(self._guess_factors) * [1.0]
        results = optimize.least_squares(self.residuals, x0=guess, args=(x, y, z, fiber_groups), bounds=(bound_lower, bound_upper))
        self._coeffs, self._factors = self._unpack_pars(results.x)
        return results

    def plot(self, x, y, z, fiber_groups, slitmap, axs=None):
        if axs is None:
            _, axs = plt.subplots(1, 5, figsize=(14,3), sharex=True, sharey=True, layout="constrained")
        gradient_model = self.ifu_gradient(self._coeffs, x, y)
        factors_model = self.ifu_factors(self._factors, fiber_groups)
        plot_gradient_fit(slitmap, z, gradient_model=gradient_model, factors_model=factors_model, telescope="Sci", axs=axs)


class IFURadialGradient:
    def __init__(self, guess_coeffs, profile="poly",
                 guess_center=(0.0, 0.0), guess_ab=(1.0, 1.0), guess_theta=0.0,
                 fix_coeffs=None, fit_geometry=False):
        self.profile = profile.lower()
        self._guess_coeffs = list(guess_coeffs)
        self._ncoeffs = len(self._guess_coeffs)
        self._fixed_coeffs = numpy.zeros_like(self._guess_coeffs, dtype=bool)
        if fix_coeffs:
            self._fixed_coeffs[fix_coeffs] = True

        self._guess_center = guess_center
        self._guess_ab = guess_ab
        self._guess_theta = guess_theta
        self.fit_geometry = fit_geometry

        self._coeffs = copy(self._guess_coeffs)
        self._center = guess_center
        self._ab = guess_ab
        self._theta = guess_theta

    @classmethod
    def elliptical_radius(cls, x, y, xc, yc, a, b, theta):
        dx = x - xc
        dy = y - yc

        cos_t = numpy.cos(theta)
        sin_t = numpy.sin(theta)

        x_prime = dx * cos_t + dy * sin_t
        y_prime = -dx * sin_t + dy * cos_t

        return numpy.sqrt((x_prime / a)**2 + (y_prime / b)**2)

    @classmethod
    def radial_gradient(cls, coeffs, x, y, center, ab, theta, profile="poly", normalize=True):
        r = cls.elliptical_radius(x, y, *center, *ab, theta)

        if profile == "poly":
            R = numpy.vstack([r**i for i in range(len(coeffs))]).T
            model = bn.nansum(R * numpy.asarray(coeffs)[None, :], axis=1)

        elif profile == "exp":
            A, k = coeffs
            model = A * numpy.exp(-k * r)

        elif profile == "power":
            A, alpha = coeffs
            with numpy.errstate(divide="ignore", invalid="ignore"):
                model = A * r**alpha
                model[numpy.isnan(model)] = 0.0

        else:
            raise ValueError(f"Unknown profile: {profile}")

        return model / bn.nanmean(model) if normalize else model

    def _pack_pars(self, coeffs=None, center=None, ab=None, theta=None):
        coeffs = numpy.where(self._fixed_coeffs, self._guess_coeffs, copy(coeffs or self._coeffs)).tolist()
        extras = []
        if self.fit_geometry:
            extras = list(center or self._center) + list(ab or self._ab) + [theta if theta is not None else self._theta]
        return coeffs + extras

    def _unpack_pars(self, pars):
        coeffs = numpy.where(self._fixed_coeffs, self._guess_coeffs, pars[:self._ncoeffs])
        if self.fit_geometry:
            xc, yc, a, b, theta = pars[self._ncoeffs:]
            return coeffs, (xc, yc), (a, b), theta
        return coeffs, self._center, self._ab, self._theta

    def __call__(self, x, y, coeffs=None, center=None, ab=None, theta=None):
        _coeffs, _center, _ab, _theta = self._unpack_pars(self._pack_pars(coeffs, center, ab, theta))
        return self.radial_gradient(_coeffs, x, y, _center, _ab, _theta, profile=self.profile)

    def residuals(self, pars, x, y, z, sigma=None):
        self._coeffs, self._center, self._ab, self._theta = self._unpack_pars(pars)
        model = self(x, y)
        res = (model - z)
        return res

    def fit(self, x, y, z, sigma=None):
        guess = self._pack_pars()
        lower = [-numpy.inf] * self._ncoeffs
        upper = [+numpy.inf] * self._ncoeffs

        if self.fit_geometry:
            xc, yc = self._guess_center
            a, b = self._guess_ab

            lower += [xc - 10, yc - 10, 0.1, 0.1, -numpy.pi]
            upper += [xc + 10, yc + 10, 10.0, 10.0, numpy.pi]

        results = optimize.least_squares(
            self.residuals,
            x0=guess,
            args=(x, y, z, sigma),
            bounds=(lower, upper),
            loss="cauchy"
        )
        self._coeffs, self._center, self._ab, self._theta = self._unpack_pars(results.x)
        return results

    def plot(self, x, y, z, slitmap, axs=None):
        if axs is None:
            _, axs = plt.subplots(1, 3, figsize=(14,3), sharex=True, sharey=True, layout="constrained")
        gradient_model = self(x, y)
        plot_radial_gradient_fit(slitmap, z, gradient_model=gradient_model, telescope="Sci", axs=axs)


class Profile1D:
    PARNAMES = ...

    def __call__(self, x):
        ...

    @classmethod
    def eval(cls, x, pars):
        instance = cls(pars, fixed={}, bounds={})
        return instance._pixelate(x)

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        self._pars = pars
        self._fixed = fixed

        self._nprofiles = self._check_sizes()
        self._npars = len(self._pars)
        self._nfixed = len(self._fixed)
        self._ignore_nans = ignore_nans
        self._oversampling_factor = oversampling_factor

        # get guess and boundaries as requested by Scipy
        self._bounds = self._parse_boundaries(self._pars, bounds)
        self._guess = self._parse_guess(self._pars)
        self._guess = numpy.clip(self._guess, *self._bounds)
        self._valid_pars = self._check_valid(self._guess, self._bounds)
        self._nfitted = self._valid_pars.sum()

    def _fwhms(self, x):
        centroids = self._pars.get("centroids", self._fixed.get("centroids"))
        half_max = self(centroids) / 2
        if x is None:
            fwhms = numpy.ones_like(centroids) * numpy.nan
            errors = numpy.ones_like(centroids) * numpy.nan
            masks = numpy.ones_like(centroids, dtype="bool")
            return fwhms, errors, masks

        x_os = self._oversample_x(x)
        models = self(x_os, collapse=False)

        indices = numpy.asarray([numpy.where(model >= half_max[i])[0][[0,-1]] if numpy.isfinite(model).all() else [0, 0] for i, model in enumerate(models)])
        fwhms = numpy.diff(x_os[indices], axis=1)
        errors = numpy.ones_like(fwhms) / self._oversampling_factor
        masks = (fwhms == 0) | (~numpy.isfinite(fwhms))

        fwhms[masks] = numpy.nan
        errors[masks] = numpy.nan

        return fwhms, errors, masks

    def _oversample_x(self, x, oversampling_factor=None):
        return oversample(x, oversampling_factor or self._oversampling_factor)

    def _pixelate(self, x, models, oversampling_factor=None):
        return pixelate(x, models, oversampling_factor or self._oversampling_factor)

    def _to_list(self, x):
        return [x[name] for name in self.PARNAMES if name in x]

    def _check_sizes(self):
        pars_all = {}
        pars_all.update(self._pars)
        pars_all.update(self._fixed)

        pars_list = self._to_list(pars_all)
        par_sizes = numpy.asarray([par.size for par in pars_list], dtype="int")
        if (par_sizes[0] != par_sizes[1:]).any():
            raise ValueError(f"Incompatible free parameter sizes: {par_sizes}")
        return par_sizes[0]

    def _check_valid(self, guess, bounds):
        bounds_valid = ~numpy.isnan(bounds)
        lower_valid = bounds_valid[0].reshape((-1, self._nprofiles))
        upper_valid = bounds_valid[1].reshape((-1, self._nprofiles))
        if not lower_valid.all() and not self._ignore_nans:
            raise ValueError(f"Invalid values in lower bounds:\n  {bounds[0]}")
        if not upper_valid.all() and not self._ignore_nans:
            raise ValueError(f"Invalid values in upper bounds:\n  {bounds[1]}")

        guess_valid = numpy.isfinite(guess).reshape((-1, self._nprofiles))
        if not guess_valid.all() and not self._ignore_nans:
            raise ValueError(f"Invalid values in guess parameters:\n  {guess}")

        fixed_valid = numpy.ones_like(guess_valid, dtype="bool")
        if len(self._fixed) != 0:
            fixed_list = self._to_list(self._fixed)
            fixed_valid = numpy.isfinite(numpy.concatenate(fixed_list)).reshape((-1, self._nprofiles))
            if not fixed_valid.all() and not self._ignore_nans:
                raise ValueError(f"Invalid values in fixed parameters:\n  {self._fixed}")

        valid_pars = guess_valid.all(0) & lower_valid.all(0) & upper_valid.all(0) & fixed_valid.all(0)
        return valid_pars

    def _parse_guess(self, guess):
        guess_list = self._to_list(guess)
        guess = numpy.concatenate(guess_list)
        return guess

    def _parse_boundaries(self, pars, bounds):

        bounds_lower, bounds_upper = [], []
        _ = numpy.ones(self._nprofiles)
        def _set_boundaries(x, x_bound):
            kind, x_range = x_bound.get("kind"), x_bound.get("range")
            if kind == "relative":
                lower = x + x_range[0]
                upper = x + x_range[1]
            elif kind == "absolute":
                lower = _ * x_range[0]
                upper = _ * x_range[1]
            else:
                lower = _ * -numpy.inf
                upper = _ * +numpy.inf

            bounds_lower.append(lower)
            bounds_upper.append(upper)

        for name in pars:
            _set_boundaries(pars.get(name), bounds.get(name, {}))

        bounds_lower = numpy.concatenate(bounds_lower)
        bounds_upper = numpy.concatenate(bounds_upper)
        bounds = numpy.asarray([bounds_lower, bounds_upper])
        return bounds

    def _validate_uncertainties(self, sigma):
        sigma = sigma if sigma is not None else 1.0
        if numpy.isnan(sigma).any():
            raise ValueError(f"Errors have non-valid values: {sigma}")
        return sigma

    def _validate_pars_scales(self, scales):
        if scales is not None and not numpy.isfinite(scales).all():
            raise ValueError(f"Invalid values in `pars_scales`: {scales}. Expected finite values")
        if scales is None:
            scales = numpy.abs(numpy.concatenate([numpy.full_like(par, numpy.nanmean(par)) for _, par in self._pars.items()]))
            scales[(scales==0)|~numpy.isfinite(scales)] = 1.0
        return scales

    def _select_fitting_mode(self, mode, x, y, sigma, pars_scales, *args, **kwargs):
        selection = numpy.tile(self._valid_pars, self._npars)
        if mode == "lsq":
            result = optimize.least_squares(self.residuals, x0=self._guess[selection], bounds=self._bounds[:, selection], x_scale=pars_scales[selection], args=(x, y, sigma), **kwargs)
        elif mode == "custom_cost":
            args_ = (x, y, sigma) + args
            fun = getattr(self, "cost_function", None)
            if fun is None:
                raise ValueError(f"Invalid value for `fun`: {fun}. Expected a callable with signature '{inspect.signature(self.residuals)}'")
            result = optimize.minimize(fun, x0=self._guess[selection], bounds=self._bounds[:, selection].T, args=args_, **kwargs)
        return result

    def _calc_covariance(self, result):

        _ = numpy.full((self._nfitted,self._nfitted), numpy.nan)

        # TODO: there are some cases where the Jacobian is a dense matrix object when mode="custom_cost"
        J = result.jac
        H_inv = getattr(result, "hess_inv", None)
        if J.ndim == 1 and H_inv is None:
            return _
        elif J.ndim == 1:
            cov = H_inv.todense()
            return cov

        try:
            cov = numpy.linalg.inv(J.T @ J)
        except numpy.linalg.LinAlgError:
            try:
                cov = numpy.linalg.pinv(J.T @ J)
            except Exception as e:
                warnings.warn(f"while calculating variance with numpy.linalg.pinv: {e}")
                cov = numpy.full((self._nfitted,self._nfitted), numpy.nan)
        return cov

    def _parse_result(self, result=None):
        if result is None:
            mask = self.pack_params(numpy.ones(self._nfitted, dtype="bool"))
            pars = self.pack_params(numpy.full(self._nfitted, numpy.nan))
            errs = self.pack_params(numpy.full(self._nfitted, numpy.nan))
            cov = numpy.full((self._nfitted,self._nfitted), numpy.nan)
            return mask, pars, errs, cov

        cov = self._calc_covariance(result)
        pars = result.x
        errs = numpy.sqrt(numpy.diag(cov))
        mask = getattr(result, "active_mask", numpy.zeros_like(pars)) != 0
        selection = numpy.tile(self._valid_pars, self._npars)
        mask |= pars <= self._bounds[0, selection]
        mask |= pars >= self._bounds[1, selection]
        # mask |= pars == self._guess[selection]
        pars[mask] = numpy.nan
        errs[mask] = numpy.nan

        mask = self.pack_params(mask)
        pars = self.pack_params(pars)
        errs = self.pack_params(errs)
        return mask, pars, errs, cov

    def unpack_params(self):
        params = [self._pars.get(name, self._fixed.get(name, None)) for name in self.PARNAMES]
        return params

    def pack_params(self, pars):
        params = {name: numpy.full(self._nprofiles, numpy.nan) for name in self._pars}
        for i, name in enumerate(self._pars.keys()):
            params[name][self._valid_pars] = pars[i*self._nfitted:(i+1)*self._nfitted]
        return params

    @update_params
    def residuals(self, pars, x, y, sigma, *args, **kwargs):
        model = self(x)
        return (model - y) / sigma

    @update_params
    def cost_function(self, pars, x, y, sigma, readnoise, collapse=True):
        rn_sq = readnoise ** 2
        model = self(x)
        loglik = (y + rn_sq) * numpy.log(model + rn_sq) - model
        if collapse:
            # print(f"{readnoise = }")
            # print(f"{y = }")
            # print(f"{sigma = }")
            # print(f"{model = }")
            # print(f"{loglik = }")
            return numpy.nansum(loglik)
        return loglik

    def chi_sq(self, x, y, sigma, collapse=True):
        dof = max(1, x.size - self._npars * self._nfitted)
        model = self(x)
        chisq = (model - y)**2 / sigma**2 / dof
        if collapse:
            return bn.nansum(chisq)
        return chisq

    def fit(self, x, y, sigma, *args, pars_scales=None, mode="lsq", **kwargs):

        sigma_ = self._validate_uncertainties(sigma)
        pars_scales_ = self._validate_pars_scales(pars_scales)

        try:
            result = self._select_fitting_mode(mode, x, y, sigma_, pars_scales_, *args, **kwargs)
        except Exception as e:
            warnings.warn(f"{e}")
            warnings.warn("data points:")
            warnings.warn(f"  {x       = }")
            warnings.warn(f"  {y       = }")
            warnings.warn(f"  {sigma_  = }")
            warnings.warn(f"  {self(x) = }")
            warnings.warn("current parameters:")
            warnings.warn(f"  guess       = {self._guess}")
            warnings.warn(f"  lower bound = {self._bounds[0]}")
            warnings.warn(f"  upper bound = {self._bounds[1]}")
            self._mask, self._pars, self._errs, self._cov = self._parse_result()
            return

        self._mask, self._pars, self._errs, self._cov = self._parse_result(result)

    def plot_residuals(self, x, y=None, sigma=None, mask=None, axs=None):
        residuals = None
        model = self(x)
        if y is not None and sigma is not None:
            residuals = (model - y) / sigma
        elif y is not None:
            residuals = model - y
        if residuals is None:
            return

        if axs is None:
            _, ax = create_subplots(to_display=True, figsize=(15,5), layout="constrained")
            axs = {"res": ax}
        elif isinstance(axs, plt.Axes):
            axs = {"res": axs}
        elif isinstance(axs, dict) and "mod" in axs and "res" not in axs:
            axs["mod"].tick_params(labelbottom=False)

            ax_divider = make_axes_locatable(axs["mod"])
            ax_res = ax_divider.append_axes("bottom", size="30%", pad="5%")
            ax_res.sharex(axs["mod"])
            axs["res"] = ax_res
        elif isinstance(axs, dict) and "res" in axs:
            pass

        axs["res"].axhline(ls="--", lw=1, color="0.2")
        axs["res"].axhline(-1.0, ls=":", lw=1, color="0.2")
        axs["res"].axhline(+1.0, ls=":", lw=1, color="0.2")
        if mask is not None:
            axs["res"].vlines(x[mask], *axs["res"].get_ylim(), lw=1, color="0.7", alpha=0.5, zorder=-1)
        axs["res"].step(x, residuals, lw=1, color="tab:blue", where="mid")
        return axs

    def plot(self, x, y=None, sigma=None, mask=None, axs=None):
        if axs is None:
            _, axs = create_subplots(to_display=True, figsize=(15,7), layout="constrained")
        if not isinstance(axs, dict) or "mod" not in axs:
            axs = {"mod": axs}

        model, model_os, x_os = self(x, return_all=True)

        if y is not None and sigma is not None:
            axs["mod"].errorbar(x, y, yerr=sigma, fmt=".-", ms=7, mew=0, lw=1, elinewidth=1, mfc="tab:red", color="0.2", ecolor="0.2")
        elif y is not None:
            axs["mod"].step(x, y, lw=1, color="0.2", where="mid")

        if any(axs["mod"].get_lines()):
            ylims = axs["mod"].get_ylim()
        else:
            ylims = None

        if mask is not None:
            axs["mod"].vlines(x[mask], *ylims, lw=1, color="0.7", alpha=0.5, zorder=-1)

        axs["mod"].step(x, model, lw=1, color="tab:blue", where="mid")
        axs["mod"].plot(x_os, model_os, "--", lw=1, color="tab:blue", alpha=0.5)

        if ylims is not None:
            axs["mod"].set_ylim(*ylims)

        self.plot_residuals(x, y, sigma, mask, axs=axs)
        return axs


class MexHatGaussians(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas",
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50, fiber_radius=1.4):
        super().__init__(pars, fixed, bounds, ignore_nans=ignore_nans, oversampling_factor=oversampling_factor)

        self._fiber_radius = fiber_radius

    def __call__(self, x, collapse=True, return_all=False):
        counts, centroids, sigmas = self.unpack_params()

        of = self._oversampling_factor
        x_os = self._oversample_x(x, oversampling_factor=of)
        dx_os = x_os[1] - x_os[0]

        # since the fiber profile has a fixed projected radius, I'm using this as the convolution kernel
        x_kernel = numpy.arange(0, 2*self._fiber_radius + dx_os, dx_os)
        kernel = fiber_profile(centroids=self._fiber_radius, radii=self._fiber_radius, x=x_kernel)
        psfs = gaussians((counts, centroids, sigmas), x_os, alpha=2.0, collapse=False)

        profiles = fftconvolve(psfs, kernel, mode="same", axes=1)
        profiles /= integrate.trapezoid(profiles, x_os, axis=1)[:, None]
        profiles *= counts[:, None]

        # pixelate models
        models = self._pixelate(x_os, profiles)

        if return_all:
            if collapse:
                return bn.nansum(models, 0), bn.nansum(profiles, 0), x_os
            return models, profiles, x_os
        if collapse:
            return bn.nansum(models, 0)
        return models


class TopHatGaussians(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50, fiber_width=1.2):
        super().__init__(pars, fixed, bounds, ignore_nans=ignore_nans, oversampling_factor=oversampling_factor)

        self._fiber_width = fiber_width

    def __call__(self, x):
        counts, centroids, sigmas = self.unpack_params()

        x_os = self._oversample_x(x)

        width = int(self._fiber_width * self._oversampling_factor)
        gaussians_ = gaussians((counts, centroids, sigmas), x_os, collapse=False)
        tophats = numpy.ones((counts.size, width)) / width
        gaussians_tophats = fftconvolve(gaussians_, tophats, mode="same", axes=1)

        model = self._pixelate(x_os, gaussians_tophats)
        model = bn.nansum(gaussians_tophats, axis=0)

        return model


class NormalGaussians(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50, alpha=2.3):
        super().__init__(pars, fixed, bounds, ignore_nans=ignore_nans, oversampling_factor=oversampling_factor)

        self._alpha = alpha

    def __call__(self, x):
        pars = self.unpack_params()

        x_os = self._oversample_x(x)
        models = gaussians(pars, x_os, alpha=self._alpha, collapse=False)
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model


class SkewedGaussians(Profile1D):

    PARNAMES = (
        "counts",    # Integral of the gaussian
        "centroids", # Mode
        "sigmas",    # Standard deviation
        "alphas"     # Shape parameter (not to be confused with skewness)
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        super().__init__(pars, fixed, bounds, ignore_nans=ignore_nans, oversampling_factor=oversampling_factor)

    def __call__(self, x):
        counts, centroids, sigmas, alphas = self.unpack_params()
        # convert to PDF parameters
        deltas = self._deltas(alphas)
        scales = self._sigma_to_scale(sigmas, deltas)
        locations = self._centroid_to_location(centroids, scales, alphas, deltas)

        # evaluate PDF shape
        x_os = self._oversample_x(x)
        shape = self._skewed_gaussian_shapes(x_os, locations, scales, alphas)
        # calculate normalization
        norms = numpy.trapz(shape, x_os, axis=1)

        models = counts[:, numpy.newaxis] * shape / norms[:, numpy.newaxis]
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model

    def _deltas(self, alphas):
        return alphas / numpy.sqrt(1 + alphas**2)

    def _m0(self, alphas, deltas):
        return skew_factor * deltas - (1-numpy.pi*0.25)*(skew_factor*deltas)**3/(1-2/numpy.pi*deltas**2) - numpy.sign(alphas)*0.5*numpy.exp(-2*numpy.pi/numpy.abs(alphas))

    def _location_to_centroid(self, locations, scales, alphas, deltas):
        m0 = self._m0(alphas, deltas)
        return locations + m0*scales

    def _centroid_to_location(self, centroids, scales, alphas, deltas):
        m0 = self._m0(alphas, deltas)
        return centroids - m0*scales

    def _scale_to_sigma(self, scales, deltas):
        return scales * numpy.sqrt(1 - (2 * deltas**2) / numpy.pi)

    def _sigma_to_scale(self, sigmas, deltas):
        denom = numpy.sqrt(1 - (2 * deltas**2) / numpy.pi)
        return sigmas / denom

    def _skewed_gaussian_shapes(self, x, locations, scales, alphas):
        z = (x[numpy.newaxis, :] - locations[:, numpy.newaxis]) / scales[:, numpy.newaxis]
        return numpy.exp(-0.5 * z**2) * (1 + special.erf(alphas[:, numpy.newaxis] * z / numpy.sqrt(2)))


class PolyGaussians(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas",
        "a",
        "b",
        "c",
        "d"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        super().__init__(pars, fixed, bounds, ignore_nans, oversampling_factor=oversampling_factor)

    def __call__(self, x):
        counts, centroids, sigmas, a, b, c, d = self.unpack_params()

        x_os = self._oversample_x(x)
        gauss = gaussians((counts, centroids, sigmas), x_os, collapse=False)
        poly = a[:,None] + b[:,None]*x_os[None,:] + c[:,None]*x_os[None,:]**2 + d[:,None]*x_os[None,:]**3
        models = gauss + poly
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model

    def _polynomial(self, x):
        counts, centroids, sigmas, a, b, c, d = self.unpack_params()
        poly = bn.nansum(a[:,None] + b[:,None]*x[None,:] + c[:,None]*x[None,:]**2 + d[:,None]*x[None,:]**3, axis=0)
        return poly


class Moffats(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas",
        "betas"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        super().__init__(pars, fixed, bounds, ignore_nans, oversampling_factor=oversampling_factor)

    def __call__(self, x):
        counts, centroids, sigmas, betas = self.unpack_params()

        x_os = self._oversample_x(x)
        moffats_ = numpy.asarray([Moffat1D(1.0, centroid, sigma, beta)(x_os) for centroid, sigma, beta in zip(centroids, sigmas, betas)])
        norms = integrate.trapezoid(moffats_, x_os, axis=1)

        models = counts[:, None] * moffats_ / norms[:, None]
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model

class Lorentzs(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        super().__init__(pars, fixed, bounds, ignore_nans=ignore_nans, oversampling_factor=oversampling_factor)

    def __call__(self, x):
        counts, centroids, sigmas = self.unpack_params()
        fwhms = sigmas * 2.354

        x_os = self._oversample_x(x)
        models = [count * Lorentz1D(2/(numpy.pi*fwhm), centroid, fwhm)(x_os) for count, centroid, fwhm in zip(counts, centroids, fwhms)]
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model

class Voigts(Profile1D):

    PARNAMES = (
        "counts",
        "centroids",
        "sigmas_l",
        "sigmas_g"
    )

    def __init__(self, pars, fixed, bounds, ignore_nans=True, oversampling_factor=50):
        super().__init__(pars, fixed, bounds, ignore_nans, oversampling_factor=oversampling_factor)

    def __call__(self, x):
        counts, centroids, sigmas_l, sigmas_g = self.unpack_params()
        fwhms_l = sigmas_l * 2.354
        fwhms_g = sigmas_g * 2.354

        x_os = self._oversample_x(x)
        models = [count * Voigt1D(centroid, 2/(numpy.pi*fwhm_l), fwhm_l, fwhm_g, method="Scipy")(x) for count, centroid, fwhm_l, fwhm_g in zip(counts, centroids, fwhms_l, fwhms_g)]
        models = self._pixelate(x_os, models)

        model = bn.nansum(models, axis=0)
        return model


PROFILES = {
    "mexhat": MexHatGaussians,
    "tophat": TopHatGaussians,
    "normal": NormalGaussians,
    "skewed": SkewedGaussians,
    "poly": PolyGaussians,
    "moffat": Moffats,
    "lorentz": Lorentzs,
    "voigt": Voigts}


class SpectralResolution(object):
    def __init__(self, res=None):
        self._res = res
        self._inter = None

    def getRes(self, wave):
        if self._inter is None:
            return self._res / 2.354
        else:
            return self._inter(wave) / 2.354

    def readFile(self, name):
        temp = open(name, "r")
        lines = temp.readlines()
        wave_res = numpy.zeros(len(lines))
        res = numpy.zeros(len(lines))
        for i in range(len(lines)):
            line = lines[i].split()
            wave_res[i] = float(line[0])
            res[i] = float(line[1])
        self._inter = interpolate.interp1d(wave_res, res)


class fit_profile1D(object):
    def __init__(self, par, func, guess_par=None, args=None):
        self._par = par
        self._func = func
        self._guess_par = guess_par
        self._args = args

    def __call__(self, x):
        return self._func(x)

    def fix_guess(self, bounds):
        guess = numpy.asarray(self._par)
        bounds = numpy.asarray(bounds)

        guess = numpy.clip(guess, *bounds)
        return guess

    def getPar(self):
        return self._par

    def res(self, par, x, y, sigma=1.0):
        self._par = par
        return (y - self(x)) / sigma

    def residuum(self, par, x, y, sigma=1.0):
        self._par = par
        return bn.nansum(((y - self(x)) / sigma) ** 2)

    def chisq(self, x, y, sigma=1.0):
        return bn.nansum(((y - self(x)) / sigma) ** 2)

    def fit(
        self,
        x,
        y,
        sigma=1.0,
        p0=None,
        bounds=(-numpy.inf, numpy.inf),
        ftol=1e-8,
        xtol=1e-8,
        maxfev=9999,
        solver="trf",
        loss="linear"
    ):
        if numpy.isnan(sigma).any():
            raise ValueError(f"Errors have non-valid values: {sigma}")
        if p0 is None and p0 is not False and self._guess_par is not None:
            self._guess_par(x, y)

        p0 = self.fix_guess(bounds)
        n = len(p0)

        # if numpy.any(numpy.isnan(p0)):
        #     raise ValueError(f"Invalid values in guess parameters:\n  {p0}")
        if numpy.any(numpy.isnan(bounds[0])):
            raise ValueError(f"Invalid values in lower bounds:\n  {bounds[0]}")
        if numpy.any(numpy.isnan(bounds[1])):
            raise ValueError(f"Invalid values in lower bounds:\n  {bounds[1]}")

        try:
            model = optimize.least_squares(
                self.res, x0=p0, bounds=bounds, args=(x, y, sigma), max_nfev=maxfev, ftol=ftol, xtol=xtol,
                method=solver, loss=loss)
        except Exception as e:
            warnings.warn(f"{e}")
            warnings.warn("data points:")
            warnings.warn(f"  {x       = }")
            warnings.warn(f"  {y       = }")
            warnings.warn(f"  {sigma   = }")
            warnings.warn(f"  {self(x) = }")
            warnings.warn("current parameters:")
            warnings.warn(f"  guess       = {p0}")
            warnings.warn(f"  lower bound = {bounds[0]}")
            warnings.warn(f"  upper bound = {bounds[1]}")
            self._par = numpy.full(n, numpy.nan)
            self._cov = numpy.full((n,n), numpy.nan)
            self._err = numpy.full(n, numpy.nan)
            self._mask = numpy.ones(self._par.size, dtype="bool")
            return

        self._par = model.x
        try:
            self._cov = numpy.linalg.inv(model.jac.T @ model.jac)
        except numpy.linalg.LinAlgError:# as e:
            # warnings.warn(f"while calculating covariance matrix: {e}. Trying numpy.linalg.pinv")
            try:
                self._cov = numpy.linalg.pinv(model.jac.T @ model.jac)
            except Exception as e:
                warnings.warn(f"while calculating variance with numpy.linalg.pinv: {e}")
                self._cov = numpy.full((n,n), numpy.nan)

        self._err = numpy.sqrt(numpy.diag(self._cov))

        self._mask = model.active_mask!=0
        self._par[self._mask] = numpy.nan
        self._err[self._mask] = numpy.nan

    def plot_residuals(self, x, y=None, sigma=None, mask=None, axs=None):
        residuals = None
        if y is not None and sigma is not None:
            residuals = (self(x) - y) / sigma
        elif y is not None:
            residuals = self(x) - y
        if residuals is None:
            return

        if axs is None:
            _, ax = create_subplots(to_display=True, figsize=(15,5), layout="constrained")
            axs = {"res": ax}
        elif isinstance(axs, plt.Axes):
            axs = {"res": axs}
        elif isinstance(axs, dict) and "mod" in axs and "res" not in axs:
            axs["mod"].tick_params(labelbottom=False)

            ax_divider = make_axes_locatable(axs["mod"])
            ax_res = ax_divider.append_axes("bottom", size="20%", pad="5%")
            ax_res.sharex(axs["mod"])
            axs["res"] = ax_res
        elif isinstance(axs, dict) and "res" in axs:
            pass

        axs["res"].axhline(ls="--", lw=1, color="0.2")
        axs["res"].axhline(-1.0, ls=":", lw=1, color="0.2")
        axs["res"].axhline(+1.0, ls=":", lw=1, color="0.2")
        axs["res"].step(x, residuals, lw=1, color="tab:blue", where="mid")
        axs["res"].vlines(x[mask], *axs["res"].get_ylim(), lw=1, color="0.7", alpha=0.5, zorder=-1)
        return axs

    def plot(self, x, y=None, sigma=None, mask=None, axs=None):
        if axs is None:
            _, axs = create_subplots(to_display=True, figsize=(15,7), layout="constrained")
        if not isinstance(axs, dict) or "mod" not in axs:
            axs = {"mod": axs}

        model = self(x)

        if y is not None and sigma is not None:
            axs["mod"].errorbar(x, y, yerr=sigma, fmt=".-", ms=7, mew=0, lw=1, elinewidth=1, mfc="tab:red", color="0.2", ecolor="0.2")
        elif y is not None:
            axs["mod"].step(x, y, lw=1, color="0.2", where="mid")

        if any(axs["mod"].get_lines()):
            ylims = axs["mod"].get_ylim()
        else:
            ylims = None

        if mask is not None:
            axs["mod"].vlines(x[mask], *ylims, lw=1, color="0.7", alpha=0.5, zorder=-1)

        axs["mod"].step(x, model, lw=1, color="tab:blue", where="mid")
        if ylims is not None:
            axs["mod"].set_ylim(*ylims)

        self.plot_residuals(x, y, sigma, mask, axs=axs)
        return axs


class fit_profile2D(object):
    def __init__(self, par, func, guess_par=None, args=None):
        self._par = par
        self._func = func
        self._guess_par = guess_par
        self._args = None

    def __call__(self, x, y):
        return self._func(x, y)

    def res(self, par, x, y, z, sigma=None, args=None):
        self._par = par
        if sigma is None:
            return z - self(x, y)
        else:
            return (z - self(x, y)) / sigma

    def fit(self, x, y, z, sigma=None, p0=None, ftol=1e-4, xtol=1e-4, warning=True):
        if p0 is None and self._guess_par is not None:
            self._guess_par(x, y, z)
        p0 = self._par

        try:
            model = optimize.leastsq(
                self.res, p0, (x, y, z, sigma), maxfev=9999, ftol=ftol, xtol=xtol
            )  # , factor=100)#, Dfun=dev_gaussian, col_deriv=True)
        except TypeError:
            model = optimize.leastsq(
                self.res, p0, (x, y, z, sigma), maxfev=9999, ftol=ftol, xtol=xtol
            )  # , factor=100)#, Dfun=dev_gaussian, col_deriv=True)
        self._par = model[0]


class parFile(fit_profile1D):
    def freePar(self):
        parameters = []
        for n in self._names:
            if self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] == 1:
                    parameters.append(float(self._parameters[n]["flux"]))
                if self._fixed[n]["vel"] == 1:
                    parameters.append(float(self._parameters[n]["vel"]))
                if self._fixed[n]["disp"] == 1:
                    parameters.append(float(self._parameters[n]["disp"]))
            if self._profile_type[n] == "Poly":
                for p in range(int(self._parameters[n]["order"])):
                    parameters.append(float(self._parameters[n]["coeff_%d" % (p)]))
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] == 1:
                    parameters.append(float(self._parameters[n]["scale"]))
        self._par = parameters

    def restoreResult(self):
        m = 0
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] == 1:
                    self._parameters[n]["scale"] = self._par[m]
                    m += 1
                self._parameters[n]["start_wave"] = float(
                    self._parameters[n]["start_wave"]
                )
                self._parameters[n]["end_wave"] = float(self._parameters[n]["end_wave"])
                self._parameters[n]["TemplateSpec"] = self._parameters[n][
                    "TemplateSpec"
                ]
            if self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] == 1:
                    self._parameters[n]["flux"] = self._par[m]
                    m += 1
                if self._fixed[n]["vel"] == 1:
                    self._parameters[n]["vel"] = self._par[m]
                    m += 1
                if self._fixed[n]["disp"] == 1:
                    self._parameters[n]["disp"] = self._par[m]
                    m += 1
                self._parameters[n]["restwave"] = float(self._parameters[n]["restwave"])
            if self._profile_type[n] == "Poly":
                for p in range(int(self._parameters[n]["order"])):
                    if self._fixed[n]["coeff_%d" % (p)] == 1:
                        self._parameters[n]["ceff_%d" % (p)] = self._par[m]
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] != 1:
                    try:
                        float(self._parameters[n]["scale"])
                    except ValueError:
                        line = self._parameters[n]["scale"].split(":")
                        if len(line) == 1:
                            self._parameters[n]["scale"] = self._parameters[line[0]][
                                "scale"
                            ]
                        else:
                            self._parameters[n]["scale"] = self._parameters[line[0]][
                                "scale"
                            ] * float(line[1])
            if self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] != 1:
                    try:
                        self._parameters[n]["flux"] = float(self._parameters[n]["flux"])
                    except ValueError:
                        line = self._parameters[n]["flux"].split(":")
                        if len(line) == 1:
                            self._parameters[n]["flux"] = self._parameters[line[0]][
                                "flux"
                            ]
                        else:
                            self._parameters[n]["flux"] = self._parameters[line[0]][
                                "flux"
                            ] * float(line[1])
                if self._fixed[n]["vel"] != 1:
                    try:
                        self._parameters[n]["vel"] = float(self._parameters[n]["vel"])
                    except ValueError:
                        self._parameters[n]["vel"] = self._parameters[
                            self._parameters[n]["vel"]
                        ]["vel"]
                if self._fixed[n]["disp"] != 1:
                    try:
                        self._parameters[n]["disp"] = float(self._parameters[n]["disp"])
                    except ValueError:
                        self._parameters[n]["disp"] = self._parameters[
                            self._parameters[n]["disp"]
                        ]["disp"]

    def restoreResultErr(self):
        m = 0
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] == 1:
                    self._parameters_err[n]["scale"] = self._par_err[m]
                    m += 1
            if self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] == 1:
                    self._parameters_err[n]["flux"] = self._par_err[m]

                    m += 1
                if self._fixed[n]["vel"] == 1:
                    self._parameters_err[n]["vel"] = self._par_err[m]
                    m += 1
                if self._fixed[n]["disp"] == 1:
                    self._parameters_err[n]["disp"] = self._par_err[m]
                    m += 1
            if self._profile_type[n] == "Poly":
                for p in range(int(self._parameters[n]["order"])):
                    if self._fixed[n]["coeff_%d" % (p)] == 1:
                        self._parameters_err[n]["ceff_%d" % (p)] = self._par_err[m]
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] != 1:
                    try:
                        float(self._parameters[n]["scale"])
                    except ValueError:
                        line = self._parameters[n]["scale"].split(":")
                        if len(line) == 1:
                            self._parameters_err[n]["scale"] = self._parameters_err[
                                line[0]
                            ]["scale"]
                        else:
                            self._parameters_err[n]["scale"] = self._parameters_err[
                                line[0]
                            ]["scale"] * float(line[1])
            if self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] != 1:
                    try:
                        float(self._parameters[n]["flux"])
                        self._parameters_err[n]["flux"] = 0
                    except ValueError:
                        line = self._parameters[n]["flux"].split(":")
                        if len(line) == 1:
                            self._parameters_err[n]["flux"] = self._parameters_err[
                                line[0]
                            ]["flux"]
                        else:
                            self._parameters_err[n]["flux"] = self._parameters_err[
                                line[0]
                            ]["flux"] * float(line[1])
                if self._fixed[n]["vel"] != 1:
                    try:
                        float(self._parameters[n]["vel"])
                        self._parameters_err[n]["vel"] = 0
                    except ValueError:
                        self._parameters_err[n]["vel"] = self._parameters_err[
                            self._parameters[n]["vel"]
                        ]["vel"]
                if self._fixed[n]["disp"] != 1:
                    try:
                        float(self._parameters[n]["disp"])
                        self._parameters_err[n]["disp"] = 0
                    except ValueError:
                        self._parameters_err[n]["disp"] = self._parameters_err[
                            self._parameters[n]["disp"]
                        ]["disp"]

    def guessPar(self, x, y):
        w = self._guess_window
        dx = bn.nanmedian(x[1:] - x[:-1])
        temp_y = copy(y)
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] == 1:
                    select_match = numpy.logical_and(
                        numpy.logical_and(
                            self._template_spec[n]._wave
                            >= float(self._parameters[n]["start_wave"]),
                            self._template_spec[n]._wave
                            <= float(self._parameters[n]["end_wave"]),
                        ),
                        numpy.in1d(self._template_spec[n]._wave, x),
                    )
                    scale_guess = (
                        bn.nansum(y[select_match])
                        / bn.nansum(self._template_spec[n]._data[select_match])
                        * 0.95
                    )
                    self._parameters[n]["scale"] = scale_guess
                    temp_y[select_match] = (
                        temp_y[select_match]
                        - self._template_spec[n]._data[select_match] * scale_guess
                    )
            if self._profile_type[n] == "Gauss":
                restwave = float(self._parameters[n]["restwave"])
                if self._fixed[n]["vel"] == 1:
                    vel = float(self._parameters[n]["vel"])
                    select = numpy.logical_and(
                        x > restwave * (vel / 300000.0 + 1) - w / 2.0,
                        x < restwave * (vel / 300000.0 + 1) + w / 2.0,
                    )
                    idx = numpy.argsort(temp_y[select])
                    vel = (x[select][idx[-1]] / restwave - 1) * 300000.0
                    self._parameters[n]["vel"] = vel
                if self._fixed[n]["flux"] == 1:
                    try:
                        vel = float(self._parameters[n]["vel"])
                    except ValueError:
                        vel = self._parameters[self._parameters[n]["vel"]]["vel"]
                    select = numpy.logical_and(
                        x > restwave * (vel / 300000.0 + 1) - w / 2.0,
                        x < restwave * (vel / 300000.0 + 1) + w / 2.0,
                    )
                    flux = bn.nansum(temp_y[select]) * dx
                    self._parameters[n]["flux"] = flux
                if self._fixed[n]["disp"] == 1:
                    try:
                        vel = float(self._parameters[n]["vel"])
                    except ValueError:
                        vel = self._parameters[self._parameters[n]["vel"]]["vel"]
                    select = numpy.logical_and(
                        x > restwave * (vel / 300000.0 + 1) - w / 2.0,
                        x < restwave * (vel / 300000.0 + 1) + w / 2.0,
                    )
                    try:
                        wave_z = restwave * (vel / 300000.0 + 1)
                        width = numpy.sqrt(
                            bn.nansum((temp_y[select] * (x[select] - wave_z) ** 2))
                            / (bn.nansum(temp_y[select]))
                        )
                        if (
                            width > self._spec_res.getRes(wave_z)
                            and not numpy.isnan(width)
                        ):
                            disp = (
                                numpy.sqrt(
                                    width**2 - self._spec_res.getRes(wave_z) ** 2
                                )
                                / (wave_z)
                                * 300000.0
                            )
                            self._parameters[n]["disp"] = disp
                    # else:
                    # self._parameters[n]['disp']=0.0
                    except Exception:
                        pass
        # print self._parameters
        self.freePar()

    def _profile(self, x):
        y = numpy.zeros(len(x))
        m = 0
        for n in self._names:
            if self._profile_type[n] == "TemplateScale":
                if self._fixed[n]["scale"] == 1:
                    scale = self._par[m]
                    self._parameters[n]["scale"] = self._par[m]
                    m += 1
                else:
                    try:
                        scale = float(self._parameters[n]["scale"])
                    except ValueError:
                        line = self._parameters[n]["scale"].split(":")
                        if len(line) == 1:
                            scale = self._parameters[line[0]]["scale"]
                        else:
                            scale = float(self._parameters[line[0]]["scale"]) * float(
                                line[1]
                            )
                scale_spec = self._template_spec[n]._data
                scale_wave = self._template_spec[n]._wave
                scale_spec[scale_wave <= float(self._parameters[n]["start_wave"])] = 0
                scale_spec[scale_wave >= float(self._parameters[n]["end_wave"])] = 0
                select_match = numpy.in1d(scale_wave, x)
                y += scale_spec[select_match] * scale

            elif self._profile_type[n] == "Poly":
                for p in range(int(self._parameters[n]["order"])):
                    coeff = self._par[m]
                    self._parameters[n]["coeff_%d" % (p)] = self._par[m]
                    y += coeff * (x / 1000.0) ** p
                    m += 1

            elif self._profile_type[n] == "Gauss":
                if self._fixed[n]["flux"] == 1:
                    flux = self._par[m]
                    self._parameters[n]["flux"] = self._par[m]
                    m += 1
                else:
                    try:
                        flux = float(self._parameters[n]["flux"])
                    except ValueError:
                        line = self._parameters[n]["flux"].split(":")
                        if len(line) == 1:
                            flux = self._parameters[line[0]]["flux"]
                        else:
                            flux = float(self._parameters[line[0]]["flux"]) * float(
                                line[1]
                            )
                if self._fixed[n]["vel"] == 1:
                    vel = self._par[m]
                    self._parameters[n]["vel"] = vel
                    wave = float(self._parameters[n]["restwave"]) * (vel / 300000.0 + 1)
                    m += 1
                else:
                    try:
                        vel = float(self._parameters[n]["vel"])
                    except ValueError:
                        vel = float(self._parameters[self._parameters[n]["vel"]]["vel"])
                    wave = float(self._parameters[n]["restwave"]) * (vel / 300000.0 + 1)
                if self._fixed[n]["disp"] == 1:
                    disp = self._par[m]
                    self._parameters[n]["disp"] = disp
                    m += 1
                else:
                    try:
                        disp = float(self._parameters[n]["disp"])
                    except ValueError:
                        disp = float(
                            self._parameters[self._parameters[n]["disp"]]["disp"]
                        )
                width = numpy.sqrt(
                    (disp / 300000.0 * wave) ** 2 + self._spec_res.getRes(wave) ** 2
                )
                y += (
                    flux
                    * numpy.exp(-0.5 * ((x - wave) / width) ** 2)
                    / (fact * numpy.fabs(width))
                )
        return y

    def writeTablePar(self, outfile):
        columns = []
        # if self._par_err is not None:
        self.restoreResultErr()
        self.restoreResult()
        for k1 in self._parameters.keys():
            for k2 in self._parameters[k1].keys():
                # print self._parameters[k1].keys()
                # print self._par_err
                # print k1,k2
                columns.append(
                    pyfits.Column(
                        name="%s_%s" % (k1, k2),
                        format="E",
                        array=[self._parameters[k1][k2]],
                    )
                )
                if self._par_err is not None:
                    # print self._par_err,k1,k2
                    columns.append(
                        pyfits.Column(
                            name="%s_%s_err" % (k1, k2),
                            format="E",
                            array=[self._parameters_err[k1][k2]],
                        )
                    )

        coldefs = pyfits.ColDefs(columns)
        tbhdu = pyfits.BinTableHDU.from_columns(coldefs)
        tbhdu.writeto(outfile, overwrite=True)

    def __init__(self, file, spec_res=0):
        fpar = open(file, "r")
        lines = fpar.readlines()
        self._names = []
        if isinstance(spec_res, float) or isinstance(spec_res, int):
            self._spec_res = SpectralResolution(float(spec_res))
        else:
            self._spec_res = SpectralResolution()
            self._spec_res.readFile(spec_res)

        self._profile_type = {}
        self._parameters = {}
        self._fixed = {}
        self._template_spec = {}

        par_comp = {}
        par_fix = {}
        for i in range(len(lines)):
            line = lines[i].split()
            if len(line) > 0:
                if (
                    line[0] == "Gauss:"
                    or line[0] == "Poly:"
                    or line[0] == "TemplateScale:"
                ):
                    if len(par_comp) != 0:
                        self._parameters[self._names[-1]] = par_comp
                        self._fixed[self._names[-1]] = par_fix
                        par_comp = {}
                        par_fix = {}
                    self._names.append(line[1])
                    self._profile_type[line[1]] = line[0][:-1]
                else:
                    par_comp[line[0]] = line[1]
                    if len(line) > 2:
                        par_fix[line[0]] = int(line[2])
                    else:
                        par_fix[line[0]] = 0
        self._parameters[self._names[-1]] = par_comp
        self._fixed[self._names[-1]] = par_fix
        self._parameters_err = copy(self._parameters)
        # for n in self._names:
        #     if self._profile_type[n]=='TemplateScale':
        #         spec = Spectrum1D()
        #         spec.loadFitsData(self._parameters[n]['TemplateSpec'])
        #         self._template_spec[n]=spec
        self.freePar()
        fit_profile1D.__init__(self, self._par, self._profile)


class Gaussian(fit_profile1D):
    def _profile(self, x):
        x_os = oversample(x, oversampling_factor=100)
        self._par[2] = abs(self._par[2])
        model = numpy.exp(-0.5 * ((x_os - self._par[1]) / abs(self._par[2])) ** 2) / (fact * abs(self._par[2]))
        model = pixelate(x_os, models=self._par[0] * model[None, :], oversampling_factor=100)[0]
        return model

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = bn.nanmedian(x[1:] - x[:-1])
        self._par[0] = bn.nansum(y[sel])
        self._par[1] = bn.nansum(x[sel] * y[sel]) / self._par[0]
        self._par[2] = numpy.sqrt(
            bn.nansum((y[sel] * (x[sel] - self._par[1]) ** 2)) / self._par[0]
        )
        self._par[0] *= dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)


class Gaussian_const(fit_profile1D):
    def _profile(self, x):
        x_os = oversample(x, oversampling_factor=100)
        model = numpy.exp(-0.5 * ((x_os - self._par[1]) / self._par[2]) ** 2) / (fact * self._par[2])
        model = pixelate(x_os, models=self._par[0] * model[None, :] + self._par[3], oversampling_factor=100)[0]
        return model

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = bn.nanmedian(x[1:] - x[:-1])
        ymin = bn.nanmin(y[sel])
        self._par[0] = bn.nansum(y[sel] - ymin)
        self._par[1] = bn.nansum(x[sel] * (y[sel] - ymin)) / self._par[0]
        self._par[2] = numpy.sqrt(
            bn.nansum(((y[sel] - ymin) * (x[sel] - self._par[1]) ** 2)) / (self._par[0])
        )
        self._par[3] = ymin
        self._par[0] *= dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)


class Gaussian_poly(fit_profile1D):
    def _profile(self, x):
        return self._par[0] * numpy.exp(
            -0.5 * ((x - self._par[1]) / self._par[2]) ** 2
        ) / (fact * self._par[2]) + numpy.polyval(self._par[3:], x)

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = abs(x[1] - x[0])
        self._par[0] = bn.nansum(y[sel])
        self._par[1] = bn.nansum(x[sel] * y[sel]) / self._par[0]
        self._par[2] = numpy.sqrt(
            bn.nansum((y[sel] * (x[sel] - self._par[1]) ** 2)) / self._par[0]
        )
        self._par[0] *= dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)


class Gaussians(fit_profile1D):
    def _profile(self, x):
        pars = numpy.split(self._par, 3)
        return gaussians(pars, x)

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile)


class Gaussians_cent(fit_profile1D):
    def _profile(self, x):
        pars = numpy.split(self._par, 2)
        pars = [pars[0], pars[1], self._args]
        return gaussians(pars, x)

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)

class Gaussians_centroids(fit_profile1D):
    def _profile(self, x):
        args = numpy.split(self._args, 2)
        pars = [args[0], self._par, args[1]]
        return gaussians(pars, x)

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)

class Gaussians_width(fit_profile1D):
    def _profile(self, x):
        args = numpy.split(self._args, 2)
        pars = [args[0], args[1], self._par]
        return gaussians(pars, x)

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)


class Gaussians_counts(fit_profile1D):
    def _profile(self, x):
        args = numpy.split(self._args, 2)
        pars = [self._par, args[0], args[1]]
        return gaussians(pars, x)

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)


class Gaussians_offset(fit_profile1D):
    def _profile(self, x):
        y = numpy.zeros(len(x))
        ncomp = len(self._args)
        for i in range(ncomp):
            y += (
                self._par[i + 1]
                * numpy.exp(
                    -0.5 * ((x - self._args[i] + self._par[-1]) / self._par[0]) ** 2
                )
                / (fact * self._par[0])
            )
        return y

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)


class Gauss_Hermite(fit_profile1D):
    def _profile(self, x):
        a, mean, sigma, h3, h4, h5, h6, h7, h8 = self._par
        w = (x - mean) / sigma
        H3 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 1.0))
        H4 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 0, 1.0))
        H5 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 0, 0, 1.0))
        H6 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 0, 0, 0, 1.0))
        H7 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 0, 0, 0, 0, 1.0))
        H8 = numpy.polynomial.hermite.hermval(w, (0, 0, 0, 0, 0, 0, 0, 0, 1.0))
        # H3 = (2.828427*w**3 - 4.242641*w)*0.408248
        # H4 = (4.*w**4 - 12.*w**2 + 3.)*0.204124
        y = (
            a
            * numpy.exp(-0.5 * w**2)
            * (1.0 + h3 * H3 + h4 * H4 + h5 * H5 + h6 * H6 + h7 * H7 + h8 * H8)
            / (fact * sigma)
        )
        return y

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        self._par[0] = bn.nansum(y[sel])
        self._par[1] = bn.nansum(x[sel] * y[sel]) / self._par[0]
        self._par[2] = numpy.sqrt(
            bn.nansum((y[sel] * (x[sel] - self._par[1]) ** 2)) / self._par[0]
        )
        self._par[3] = 0.0
        self._par[4] = 0.0
        self._par[5] = 0.0
        self._par[6] = 0.0
        self._par[7] = 0.0
        self._par[8] = 0.0

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)


class Moffat(fit_profile1D):
    def _profile(self, x):
        (flux, cent, fwhm, beta) = self._par
        r_d = fwhm / (2.0 * numpy.sqrt(2 ** (1.0 / beta) - 1.0))
        sigma_0 = (beta - 1.0) * flux / (numpy.pi * (r_d**2))
        y = sigma_0 * (1.0 + ((x - cent) / r_d) ** 2) ** (-beta)
        return y

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        self._par[0] = bn.nansum(y[sel])
        self._par[1] = bn.nansum(x[sel] * y[sel]) / self._par[0]
        self._par[2] = numpy.sqrt(
            bn.nansum((y[sel] * (x[sel] - self._par[1]) ** 2)) / self._par[0]
        )
        self._par[3] = 2.0

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)


class Exponential_constant(fit_profile1D):
    def _profile(self, x):
        scale, time, const = self._par
        y = scale * numpy.exp(x / time) + const
        return y

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile)


class Gaussian2D(fit_profile2D):
    def _profile(self, x, y):
        return (
            self._par[0]
            * numpy.exp(
                -0.5
                * (
                    ((x - self._par[1]) / self._par[3]) ** 2
                    + ((y - self._par[2]) / self._par[4]) ** 2
                )
            )
            / (fact * fact * self._par[3] * self._par[4])
        )

    def _guess_par(self, x, y, z):
        self._par[0] = bn.nansum(z)
        self._par[1] = bn.nansum(x * z) / self._par[0]
        self._par[2] = bn.nansum(y * z) / self._par[0]
        indcol = numpy.around(x) == numpy.around(self._par[1])
        indrow = numpy.around(y) == numpy.around(self._par[2])
        self._par[3] = numpy.sqrt(
            bn.nansum(z[indrow] * (x[indrow] - self._par[2]) ** 2)
            / bn.nansum(z[indrow])
        )
        self._par[4] = numpy.sqrt(
            bn.nansum(z[indcol] * (y[indcol] - self._par[1]) ** 2)
            / bn.nansum(z[indcol])
        )

    def __init__(self, par):
        fit_profile2D.__init__(self, par, self._profile, self._guess_par)


class LegandrePoly(object):
    def __init__(self, coeff, min_x=None, max_x=None):
        self._min_x = min_x
        self._max_x = max_x
        self._coeff = coeff
        self._poly = []
        for i in range(len(coeff)):
            self._poly.append(special.legendre(i))

    def __call__(self, x):
        y = numpy.zeros(len(x), dtype=numpy.float32)
        if self._min_x is None:
            self._min_x = bn.nanmin(x)
        if self._max_x is None:
            self._max_x = bn.nanmax(x)
        x_poly = (x - self._min_x) * 1.98 / numpy.abs(
            (numpy.abs(self._max_x) - numpy.abs(self._min_x))
        ) - 0.99
        for i in range(len(self._coeff)):
            y += self._poly[i](x_poly) * self._coeff[i]
        return y

    def fit(self, x, y):
        eigen_poly = numpy.zeros((len(x), len(self._coeff)), dtype=numpy.float32)
        for i in range(len(self._coeff)):
            self._coeff = numpy.zeros(len(self._coeff))
            self._coeff[i] = 1
            eigen_poly[:, i] = self(x)
        #        print eigen_poly, y
        self._coeff = numpy.linalg.lstsq(eigen_poly, y, rcond=None)[0]


def gaussian(p, x):
    return (
        abs(p[0]) * numpy.exp(-0.5 * ((x - p[1]) / abs(p[2])) ** 2) / (fact * abs(p[2]))
    )


def gaussian_const(p, x):
    return p[0] * numpy.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) / (fact * p[2]) + p[3]


def gaussian2d(p, x, y):
    return (
        p[0]
        * numpy.exp(-0.5 * (((x - p[1]) / p[3]) ** 2 + ((y - p[2]) / p[4]) ** 2))
        / (fact * fact * p[3] * p[4])
    )


def gaussian_poly(p, x):
    return p[0] * numpy.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) / (
        fact * p[2]
    ) + numpy.polyval(p[3:], x)


def gaussian_multi(p, x):
    y = numpy.zeros(len(x))
    ncomp = len(p) / 3
    for i in range(ncomp):
        y += (
            abs(p[i])
            * numpy.exp(-0.5 * ((x - p[i + ncomp]) / abs(p[i + 2 * ncomp])) ** 2)
            / (fact * abs(p[i + 2 * ncomp]))
        )
    return y


def gaussian_width_multi(p, x, pos):
    y = numpy.zeros(len(x))
    ncomp = len(pos)
    for i in range(ncomp):
        y += p[i + 1] * numpy.exp(-0.5 * ((x - pos[i]) / p[0]) ** 2) / (fact * p[0])
    return y


def gaussian_width_multi_offset(p, x, pos):
    y = numpy.zeros(len(x))
    ncomp = len(pos)
    for i in range(ncomp):
        y += (
            p[i + 1]
            * numpy.exp(-0.5 * ((x - pos[i] + p[-1]) / p[0]) ** 2)
            / (fact * p[0])
        )
    return y


def gauss_hermite(p, x):  # , a, mean, sigma, h3, h4):
    a, mean, sigma, h3, h4 = p
    w = (x - mean) / sigma
    H3 = (2.828427 * w**3 - 4.242641 * w) * 0.408248
    H4 = (4.0 * w**4 - 12.0 * w**2 + 3.0) * 0.204124

    return a * numpy.exp(-0.5 * w**2) * (1.0 + h3 * H3 + h4 * H4) / (fact * sigma)


def res_gaussian(p, x, y, sigma):
    return (y - gaussian(p, x)) / sigma


def res_gaussian_const(p, x, y, sigma):
    return (y - gaussian_const(p, x)) / sigma


def res_gaussian2d(p, x, y, z, sigma):
    return (z - gaussian2d(p, x, y)) / sigma


def res_gaussian_poly(p, x, y, sigma):
    return (y - gaussian_poly(p, x)) / sigma


def res_gaussian_multi(p, x, y, sigma):
    return (y - gaussian_multi(p, x)) / sigma


def res_gaussian_width_multi(p, x, y, pos, sigma):
    return (y - gaussian_width_multi(p, x, pos)) / sigma


def res_gaussian_width_multi_offset(p, x, y, pos, sigma):
    return (y - gaussian_width_multi_offset(p, x, pos)) / sigma


def res_gauss_hermite(p, x, y, sigma):
    return (y - gauss_hermite(p, x)) / sigma


def chisq_gaussian_width_multi(p, x, y, pos, sigma):
    return bn.nansum(((y - gaussian_width_multi(p, x, pos)) / sigma) ** 2)


def dev_gaussian(p, x, y, sigma):
    z = numpy.zeros((3, len(x)))

    d = x - p[1]
    e = -0.5 * ((d) / p[2]) ** 2

    z[0] = -numpy.exp(e) / (fact * p[2])
    z[1] = -p[0] * d * numpy.exp(e) / (fact * p[2] ** 3)
    z[2] = -p[0] * (d**2) * numpy.exp(e) / (fact * p[2] ** 4)

    return z / sigma


def dev_gaussian_const(p, x, y, sigma):
    z = numpy.zeros((4, len(x)))

    d = x - p[1]
    e = -0.5 * ((d) / p[2]) ** 2

    z[0] = -numpy.exp(e) / (fact * p[2])
    z[1] = -p[0] * d * numpy.exp(e) / (fact * p[2] ** 3)
    z[2] = -p[0] * (d**2) * numpy.exp(e) / (fact * p[2] ** 4)
    z[3] = -1.0

    return z / sigma


def dev_gaussian_poly1d(p, x, y, sigma):
    z = numpy.zeros((len(p), len(x)))

    d = x - p[1]
    e = -0.5 * (d / p[2]) ** 2

    z[0] = -numpy.exp(e) / (fact * p[2])
    z[1] = -p[0] * d * numpy.exp(e) / (fact * p[2] ** 3)
    z[2] = -p[0] * (d**2) * numpy.exp(e) / (fact * p[2] ** 4)
    for i in range(len(p) - 3):
        z[3 + i] = -(x ** (len(p) - 4 - i))

    return z / sigma


def dev_gaussian_width_multi(p, x, y, pos, sigma):
    ncomp = len(pos)
    z = numpy.zeros((ncomp + 1, len(x)))

    for i in range(ncomp):
        d = x - pos[i]
        e = -0.5 * ((d) / p[0]) ** 2
        z[0] += -p[i + 1] * (d**2) * numpy.exp(e) / (fact * p[0] ** 4)
        z[i + 1] = -numpy.exp(e) / (fact * p[0])

    return z / sigma


def fit_gaussian(x, y, sigma=1.0, p0=None, ftol=1e-4, xtol=1e-4, warning=True):
    sel = numpy.isfinite(y)
    dx = abs(x[1] - x[0])

    # if bn.nansum(sel)>3:
    if p0 is None or len(p0) != 3:
        p0 = numpy.zeros(3)
        p0[0] = bn.nansum(y[sel])
        p0[1] = bn.nansum(x[sel] * y[sel]) / p0[0]
        p0[2] = numpy.sqrt(bn.nansum((y[sel] * (x[sel] - p0[1]) ** 2)) / p0[0])
        p0[0] *= dx

    sol = optimize.leastsq(
        res_gaussian, p0, (x, y, sigma), maxfev=9999, ftol=ftol, xtol=xtol
    )  # , factor=100)#, Dfun=dev_gaussian, col_deriv=True)

    return sol[0]


def fit_gaussian_const(x, y, sigma=1.0, p0=None, ftol=1e-6, xtol=1e-6, warning=True):
    cdelt = abs(x[1] - x[0])
    if p0 is None or len(p0) != 4:
        p0 = numpy.zeros(4)
        ymin = bn.nanmin(y)
        p0[0] = bn.nansum(y - ymin)
        p0[1] = bn.nansum(x * (y - ymin)) / p0[0]
        p0[2] = numpy.sqrt(bn.nansum(((y - ymin) * (x - p0[1]) ** 2)) / p0[0])
        p0[3] = ymin
        p0[0] *= cdelt

    sol = optimize.leastsq(
        res_gaussian_const, p0, (x, y, sigma), maxfev=99999, ftol=ftol, xtol=xtol
    )[
        0
    ]  # , Dfun=dev_gaussian_const, col_deriv=True)

    ##print sol
    # plt.clf()
    ##plt.axvline(5577.347, ls='--', c='k')
    # plt.plot(x,y,'k',drawstyle='steps-mid')
    # plt.plot(x,gaussian_const(sol, x),'r')#,drawstyle='steps-mid')
    # plt.draw()
    ###raw_input()

    return sol


def fit_gaussian_poly(x, y, sigma=1.0, npoly=0):
    cdelt = abs(x[1] - x[0])

    p0 = numpy.zeros(3 + npoly + 1)
    p0[0] = bn.nansum(y)
    p0[1] = bn.nansum(x * y) / p0[0]
    p0[2] = numpy.sqrt(bn.nansum(y * (x - p0[1]) ** 2) / p0[0])
    p0[0] *= cdelt

    sol = optimize.leastsq(
        res_gaussian_poly, p0, (x, y, sigma)
    )  # , Dfun=dev_gaussian_poly, col_deriv=True)

    # plt.plot(x,y,'k', drawstyle='steps-mid')
    # plt.plot(x,gaussian_poly(sol[0], x),'-r')#,drawstyle='steps-mid')
    # plt.draw()
    # raw_input()
    # plt.clf()

    return sol[0]


def fit_gaussian_multi(x, y, ncomp=1, sigma=1.0, f0=[1.0], m0=[0.0], s0=[1.0]):
    p0 = numpy.zeros(3 * ncomp)
    if len(f0) == ncomp:
        p0[:ncomp] = f0
    else:
        p0[:ncomp] = f0[0]
    if len(m0) == ncomp:
        p0[ncomp : 2 * ncomp] = m0
    else:
        p0[ncomp : 2 * ncomp] = m0[0]
    if len(s0) == ncomp:
        p0[2 * ncomp : 3 * ncomp] = s0
    else:
        p0[2 * ncomp : 3 * ncomp] = s0[0]

    sol = optimize.leastsq(res_gaussian_multi, p0, (x, y, sigma))

    return sol[0]


def fit_gaussian_width_multi(x, y, pos, sigma=1.0, flux0=1.0, width0=1.0):
    ncomp = len(pos)
    p0 = numpy.zeros(ncomp + 1)
    p0[0] = width0
    p0[1 : ncomp + 1] = flux0

    sol = optimize.leastsq(res_gaussian_width_multi, p0, (x, y, pos, sigma), ftol=0.1)

    return sol[0]


def fit_gaussian_width_multi_offset(x, y, pos, sigma=1.0, flux0=1.0, width0=1.0):
    ncomp = len(pos)
    p0 = numpy.zeros(ncomp + 2)
    p0[0] = width0
    p0[1 : ncomp + 1] = flux0
    p0[-1] = 0.0

    sol = optimize.leastsq(
        res_gaussian_width_multi_offset, p0, (x, y, pos, sigma)
    )  # , ftol=0.1)

    # print sol[0][0], sol[0][-1]
    # plt.plot(x,y,'k')#,drawstyle='steps-mid')
    # plt.plot(x,gaussian_width_multi_offset(sol[0], x, pos),'r')#,drawstyle='steps-mid')
    # plt.draw()
    # raw_input()
    # plt.clf()

    return sol[0]


def fit_gaussian2d(x, y, z, sigma=1.0):
    p0 = numpy.zeros(5)
    p0[0] = bn.nansum(z)
    p0[1] = bn.nansum(x * z) / p0[0]
    p0[2] = bn.nansum(y * z) / p0[0]
    indcol = numpy.around(x) == numpy.around(p0[1])
    indrow = numpy.around(y) == numpy.around(p0[2])
    p0[3] = numpy.sqrt(
        bn.nansum(z[indrow] * (x[indrow] - p0[2]) ** 2) / bn.nansum(z[indrow])
    )
    p0[4] = numpy.sqrt(
        bn.nansum(z[indcol] * (y[indcol] - p0[1]) ** 2) / bn.nansum(z[indcol])
    )

    sol = optimize.leastsq(res_gaussian2d, p0, (x, y, z, sigma))

    return sol[0]


def fit_gauss_hermite(x, y, sigma=1.0, p0=None):
    sel = numpy.isfinite(y)

    # if bn.nansum(sel)>3:
    if p0 is None or len(p0) != 5:
        p0 = numpy.zeros(5)
        p0[0] = bn.nansum(y[sel])
        p0[1] = bn.nansum(x[sel] * y[sel]) / p0[0]
        p0[2] = numpy.sqrt(bn.nansum((y[sel] * (x[sel] - p0[1]) ** 2)) / p0[0])
        p0[3] = 0.0
        p0[4] = 0.0

    sol = optimize.leastsq(
        res_gauss_hermite, p0, (x, y, sigma), maxfev=9999, ftol=1e-9, xtol=1e-9
    )  # , factor=100)#, Dfun=dev_gaussian, col_deriv=True)

    return sol[0]
