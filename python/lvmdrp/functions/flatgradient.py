# encoding: utf-8
#
# @Author: Knox S. Long
# @Date: Nov 10, 2026
# @Filename: flatgradient
# @License: BSD 3-Clause
# @Copyright: SDSS-V LVM

"""Large-scale flat-field error of the science IFU, measured from sky lines.

In each science exposure the strength of a bright airglow line should be
nearly uniform across the science IFU, so its pattern measures the
large-scale flat-field error left in the frame (sdss/lvmdrp#268). For each
line, the line flux of every good science fiber is fitted with three
spectrograph factors times a plane across the IFU:

    z_i / <z> = f_s(i) * (1 + (gx * x_i + gy * y_i) / 1000)

with x, y the fiber positions in mm (``xpmm``, ``ypmm``), so that gx and gy
are in % per 10 mm. The fit uses :class:`lvmdrp.core.fit_profile.IFUGradient`,
with iterative clipping of outlying fibers.

The lines fitted are 5577 (b), 6364 (r) and 8399 (z). A second set of OH
lines in r and z is measured the same way but is never used by a
correction, so it can be used to test one. The spectrograph factors are also
measured in the fibers of each sky IFU, which see the factors but not the
plane across the science IFU.
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from lvmdrp import log
from lvmdrp.core.constants import get_lvmcore_version, get_lvmdrp_version
from lvmdrp.core.fit_profile import IFUGradient

# fitted lines: arm -> (line center, side bands), Angstrom
SKYLINES = {
    'b': (5577.35, ((5566.0, 5573.0), (5582.0, 5586.0))),
    'r': (6363.78, ((6350.0, 6355.0), (6366.5, 6378.0))),
    'z': (8399.18, ((8393.0, 8396.5), (8402.5, 8412.5))),
}
# half width of the window over which a fitted line is summed, Angstrom
LINE_HALF_WIDTH = 4.0
# held-out OH lines: name -> (arm, line window, side bands), Angstrom
HELDOUT_LINES = {
    'oh6864': ('r', (6861.0, 6867.0), ((6852.0, 6858.0), (6874.0, 6878.0))),
    'oh6923': ('r', (6920.0, 6926.5), ((6916.0, 6918.5), (6929.0, 6935.0))),
    'oh7316': ('r', (7313.0, 7319.5), ((7307.0, 7310.0), (7321.0, 7325.0))),
    'oh7341': ('r', (7337.5, 7344.0), ((7332.0, 7335.5), (7347.0, 7354.0))),
    'oh8344': ('z', (8341.5, 8347.5), ((8335.0, 8339.0), (8357.0, 8362.0))),
    'oh8430': ('z', (8427.0, 8433.5), ((8420.5, 8424.0), (8436.0, 8442.0))),
    'oh8465': ('z', (8462.0, 8469.0), ((8456.0, 8459.5), (8471.0, 8478.0))),
    'oh8827': ('z', (8824.0, 8830.5), ((8815.0, 8821.0), (8840.0, 8846.0))),
    'oh8886': ('z', (8882.5, 8889.0), ((8874.0, 8880.0), (8892.0, 8899.0))),
    'oh8920': ('z', (8916.0, 8923.0), ((8907.0, 8913.0), (8926.0, 8933.0))),
}
# held-out lines summed per arm
HELDOUT_GROUPS = {
    'r_oh': ('r', ['oh6864', 'oh6923', 'oh7316', 'oh7341']),
    'z_oh': ('z', ['oh8344', 'oh8430', 'oh8465', 'oh8827', 'oh8886', 'oh8920']),
}
# continuum windows used to reject stars: arm -> center, Angstrom
STAR_CONTINUUM = {'b': 4600.0, 'r': 6760.0, 'z': 8170.0}
STAR_CONTINUUM_HALF_WIDTH = 25.0
# fibers whose continuum is outside these percentiles are rejected
STAR_PERCENTILES = (5.0, 97.0)
# clipping threshold of the plane fit, in robust standard deviations
CLIP_SIGMA = 4.0
# fewest fibers for a plane fit, and per spectrograph for sky-fiber factors
MIN_FIBERS = 100
MIN_SKY_FIBERS = 5
SKY_IFUS = ('SkyE', 'SkyW')


def _window(wave: np.ndarray, lo: float, hi: float) -> slice:
    """Returns the slice of `wave` between `lo` and `hi` (inclusive)."""
    idx = np.flatnonzero((wave >= lo) & (wave <= hi))
    if idx.size == 0:
        raise ValueError(f'no pixels between {lo} and {hi} Angstrom')
    return slice(idx[0], idx[-1] + 1)


def line_flux(flux: np.ndarray, wave: np.ndarray, window: Tuple[float, float],
              sides: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Measures the flux of a sky line in each fiber.

    The flux is the sum over the line window minus the median of the side
    bands, times the pixel size.

    Args:
        flux: fluxes, fibers x wavelength.
        wave: wavelength of each pixel, Angstrom.
        window: lower and upper limit of the line window, Angstrom.
        sides: the side bands (lower and upper limit each), Angstrom.

    Returns:
        The line flux of each fiber, NaN where the window has a non-finite pixel.
    """
    line = flux[:, _window(wave, *window)].astype(float)
    side = np.hstack([flux[:, _window(wave, lo, hi)] for lo, hi in sides]).astype(float)
    with np.errstate(all='ignore'):
        cont = np.nanmedian(side, axis=1)
        z = np.nansum(line - cont[:, None], axis=1) * np.median(np.diff(wave))
    z[~np.all(np.isfinite(line), axis=1)] = np.nan
    return z


def star_mask(flux: np.ndarray, wave: np.ndarray, arm: str,
              percentiles: Tuple[float, float] = STAR_PERCENTILES,
              throughput: Optional[np.ndarray] = None) -> np.ndarray:
    """Flags fibers dominated by stars or other bright sources.

    Fibers whose median continuum near `STAR_CONTINUUM[arm]` is outside the
    given percentiles of all fibers, or not finite, are flagged. With
    `throughput`, the continuum is divided by it first, so that a large-scale
    gradient in the frame doesn't decide which fibers fall outside the
    percentiles.

    Args:
        flux: fluxes, fibers x wavelength.
        wave: wavelength of each pixel, Angstrom.
        arm: 'b', 'r' or 'z'.
        percentiles: lower and upper percentile of the continuum kept.
        throughput: relative throughput of each fiber, or None.

    Returns:
        True for the fibers to reject.
    """
    cwave = STAR_CONTINUUM[arm]
    sel = _window(wave, cwave - STAR_CONTINUUM_HALF_WIDTH, cwave + STAR_CONTINUUM_HALF_WIDTH)
    with np.errstate(all='ignore'):
        cont = np.nanmedian(flux[:, sel].astype(float), axis=1)
        if throughput is not None:
            cont = cont / np.asarray(throughput, dtype=float)
    ok = np.isfinite(cont)
    if not ok.any():
        return np.ones(flux.shape[0], dtype=bool)
    lo, hi = np.percentile(cont[ok], percentiles)
    return ~ok | (cont < lo) | (cont > hi)


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: np.ndarray,
              clip: float = CLIP_SIGMA, niter: int = 6,
              min_fibers: int = MIN_FIBERS) -> Optional[Dict[str, float]]:
    """Fits spectrograph factors and a plane to a sky-line flux across the IFU.

    The model is ``z / <z> = f_s * (1 + (gx * x + gy * y) / 1000)``, fitted with
    :class:`IFUGradient` (its cross term fixed at zero). Fibers more than
    `clip` robust standard deviations from the model are rejected and the fit
    is repeated, up to `niter` times.

    Args:
        x: fiber x positions, mm.
        y: fiber y positions, mm.
        z: line flux of each fiber (NaN for fibers not to use).
        spec: spectrograph of each fiber (1, 2 or 3).
        clip: clipping threshold, in robust standard deviations.
        niter: largest number of clipping iterations.
        min_fibers: fewest usable fibers for a fit.

    Returns:
        A dict with gx and gy (% per 10 mm), the factors f1, f2 and f3
        (normalized to a mean of 1), sp1, sp2 and sp3 (the mean of the model
        over the fibers of each spectrograph, relative to the mean over all
        fibers, %), rms (the rms residual per fiber, %) and n (the number of
        fibers fitted); None if fewer than `min_fibers` fibers are usable.
    """
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    spec = np.asarray(spec, dtype=int)
    good = np.isfinite(z) & (z > 0) & np.isfinite(x) & np.isfinite(y)
    if good.sum() < min_fibers:
        return None

    def _fit(sel):
        zn = z[sel] / np.mean(z[sel])
        model = IFUGradient([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0], fixed_coeffs=[3])
        model.fit(x[sel], y[sel], zn, spec[sel])
        return np.asarray(model._coeffs, dtype=float), np.asarray(model._factors, dtype=float)

    def _model(coeffs, factors):
        return IFUGradient.ifu_joint_model(coeffs, factors, x, y, spec, normalize=False)

    use = good.copy()
    for _ in range(niter):
        coeffs, factors = _fit(use)
        model = _model(coeffs, factors)
        res = z / np.mean(z[use]) - model / np.mean(model[use])
        mad = 1.4826 * np.median(np.abs(res[use] - np.median(res[use])))
        new = good & (np.abs(res - np.median(res[use])) < clip * mad)
        if np.array_equal(new, use) or new.sum() < min_fibers:
            break
        use = new
    coeffs, factors = _fit(use)
    model = _model(coeffs, factors)
    res = z[use] / np.mean(z[use]) - model[use] / np.mean(model[use])

    factors = factors / np.mean(factors)
    gx, gy = 1000 * coeffs[2] / coeffs[0], 1000 * coeffs[1] / coeffs[0]
    plane = factors[spec - 1] * (1 + (gx * x + gy * y) / 1000)
    sp = [100 * (np.mean(plane[use & (spec == s)]) / np.mean(plane[use]) - 1)
          if np.any(use & (spec == s)) else np.nan for s in (1, 2, 3)]
    return {'gx': gx, 'gy': gy, 'f1': factors[0], 'f2': factors[1], 'f3': factors[2],
            'sp1': sp[0], 'sp2': sp[1], 'sp3': sp[2],
            'rms': 100 * np.sqrt(np.mean(res ** 2)), 'n': int(use.sum())}


def sky_fiber_factors(z: np.ndarray, spec: np.ndarray, cont: Optional[np.ndarray] = None,
                      clip: float = CLIP_SIGMA,
                      min_fibers: int = MIN_SKY_FIBERS) -> Optional[Dict[str, float]]:
    """Measures the spectrograph factors in the fibers of one sky IFU.

    The sky is uniform across a sky IFU, so the median line flux of each
    spectrograph's fibers, relative to the mean of the three, is the
    spectrograph factor.

    Args:
        z: line flux of each fiber.
        spec: spectrograph of each fiber (1, 2 or 3).
        cont: continuum of each fiber, to reject outliers such as stars.
        clip: clipping threshold for the continuum, in robust standard deviations.
        min_fibers: fewest usable fibers per spectrograph.

    Returns:
        A dict with the factors f1, f2 and f3, ef (the typical error of a
        factor) and n (the number of fibers used); None if a spectrograph has
        fewer than `min_fibers` usable fibers.
    """
    z = np.asarray(z, dtype=float)
    spec = np.asarray(spec, dtype=int)
    ok = np.isfinite(z) & (z > 0)
    if cont is not None:
        cont = np.asarray(cont, dtype=float)
        ok &= np.isfinite(cont)
        if ok.sum() >= 3 * min_fibers:
            c = cont[ok]
            mad = 1.4826 * np.median(np.abs(c - np.median(c)))
            if mad > 0:
                ok[ok] = np.abs(c - np.median(c)) < clip * mad
    med, err = [], []
    for s in (1, 2, 3):
        zs = z[ok & (spec == s)]
        if zs.size < min_fibers:
            return None
        med.append(np.median(zs))
        err.append(1.2533 * 1.4826 * np.median(np.abs(zs - np.median(zs))) / np.sqrt(zs.size))
    mean = np.mean(med)
    return {'f1': med[0] / mean, 'f2': med[1] / mean, 'f3': med[2] / mean,
            'ef': float(np.mean(err)) / mean, 'n': int(ok.sum())}


def measure_skyline_planes(flux: np.ndarray, wave: np.ndarray, slitmap,
                           heldout: bool = True,
                           sky_fibers: bool = True) -> Dict[str, Optional[Dict[str, float]]]:
    """Measures the large-scale flat-field error of one exposure from its sky lines.

    Args:
        flux: fluxes of all fibers, fibers x wavelength (e.g. of an lvmCFrame).
        wave: wavelength of each pixel, Angstrom.
        slitmap: the slitmap table, with columns telescope, fibstatus,
            spectrographid, xpmm and ypmm.
        heldout: also measure the held-out OH lines.
        sky_fibers: also measure the spectrograph factors in the sky IFUs.

    Returns:
        A dict of results, each a dict as returned by :func:`fit_plane` or
        :func:`sky_fiber_factors` (None where the fit was not possible): 'b',
        'r' and 'z' for the fitted lines, 'r_oh' and 'z_oh' for the held-out
        lines, and '<ifu>_<arm>' (e.g. 'skye_b') for the sky-fiber factors.
    """
    wave = np.asarray(wave, dtype=float)
    telescope = np.asarray(slitmap['telescope']).astype(str)
    status = np.asarray(slitmap['fibstatus'], dtype=int)
    allspec = np.asarray(slitmap['spectrographid'], dtype=int)
    sci = np.flatnonzero((telescope == 'Sci') & (status == 0))
    x = np.asarray(slitmap['xpmm'], dtype=float)[sci]
    y = np.asarray(slitmap['ypmm'], dtype=float)[sci]
    spec = allspec[sci]
    sflux = np.asarray(flux[sci], dtype=np.float32)

    out = {}
    stars = {}
    for arm, (cwave, sides) in SKYLINES.items():
        stars[arm] = star_mask(sflux, wave, arm)
        z = line_flux(sflux, wave, (cwave - LINE_HALF_WIDTH, cwave + LINE_HALF_WIDTH), sides)
        out[arm] = fit_plane(x, y, np.where(stars[arm], np.nan, z), spec)
        if out[arm] is not None:
            # stars again, with the continuum flattened by the fit: otherwise a large gradient
            # decides which fibers fall outside the percentiles, which biases the plane
            k = fiber_throughput(np.array([out[arm][f] for f in ('f1', 'f2', 'f3')]),
                                 out[arm]['gx'], out[arm]['gy'], x, y, spec)
            stars[arm] = star_mask(sflux, wave, arm, throughput=k)
            out[arm] = fit_plane(x, y, np.where(stars[arm], np.nan, z), spec)

    if heldout:
        for name, (arm, members) in HELDOUT_GROUPS.items():
            z = np.sum([line_flux(sflux, wave, HELDOUT_LINES[m][1], HELDOUT_LINES[m][2])
                        for m in members], axis=0)
            out[name] = fit_plane(x, y, np.where(stars[arm], np.nan, z), spec)

    if sky_fibers:
        for ifu in SKY_IFUS:
            sel = np.flatnonzero((telescope == ifu) & (status == 0))
            kflux = np.asarray(flux[sel], dtype=np.float32)
            for arm, (cwave, sides) in SKYLINES.items():
                key = f'{ifu.lower()}_{arm}'
                if sel.size == 0:
                    out[key] = None
                    continue
                c0 = STAR_CONTINUUM[arm]
                csel = _window(wave, c0 - STAR_CONTINUUM_HALF_WIDTH, c0 + STAR_CONTINUUM_HALF_WIDTH)
                with np.errstate(all='ignore'):
                    cont = np.nanmedian(kflux[:, csel].astype(float), axis=1)
                z = line_flux(kflux, wave, (cwave - LINE_HALF_WIDTH, cwave + LINE_HALF_WIDTH), sides)
                out[key] = sky_fiber_factors(z, allspec[sel], cont=cont)
    return out


# header keywords: result key -> (keyword part, label), and the quantities written for each kind
_HEADER_NAMES = {'b': ('B', '5577'), 'r': ('R', '6364'), 'z': ('Z', '8399'),
                 'r_oh': ('ROH', 'r held-out OH'), 'z_oh': ('ZOH', 'z held-out OH'),
                 'skye_b': ('SKYE B', 'SkyE 5577'), 'skye_r': ('SKYE R', 'SkyE 6364'),
                 'skye_z': ('SKYE Z', 'SkyE 8399'), 'skyw_b': ('SKYW B', 'SkyW 5577'),
                 'skyw_r': ('SKYW R', 'SkyW 6364'), 'skyw_z': ('SKYW Z', 'SkyW 8399')}
_PLANE_ITEMS = (('gx', 'GX', 'plane x [%/10mm]'), ('gy', 'GY', 'plane y [%/10mm]'),
                ('f1', 'F1', 'factor sp1'), ('f2', 'F2', 'factor sp2'), ('f3', 'F3', 'factor sp3'),
                ('rms', 'RMS', 'residual rms per fiber [%]'), ('n', 'N', 'fibers fitted'))
_SKY_ITEMS = (('f1', 'F1', 'factor sp1'), ('f2', 'F2', 'factor sp2'), ('f3', 'F3', 'factor sp3'),
              ('n', 'N', 'fibers used'))


def skyline_plane_cards(results: Dict[str, Optional[Dict[str, float]]],
                        prefix: str = 'SKYPLANE') -> list:
    """Converts the results of :func:`measure_skyline_planes` to header cards.

    Keywords are ``HIERARCH <prefix> <line> <item>``, e.g.
    ``HIERARCH SKYPLANE Z GX``. Results that could not be measured are written
    as NaN (which :meth:`Header.setHdrValue` writes as its placeholder) and N 0.

    Args:
        results: the dict returned by :func:`measure_skyline_planes`.
        prefix: the first word of the keywords.

    Returns:
        A list of (keyword, value, comment) tuples.
    """
    cards = []
    for key, res in results.items():
        if key not in _HEADER_NAMES:
            continue
        name, label = _HEADER_NAMES[key]
        items = _SKY_ITEMS if key.startswith('sky') else _PLANE_ITEMS
        for item, kw, comment in items:
            if res is None:
                value = 0 if item == 'n' else np.nan
            else:
                value = int(res[item]) if item == 'n' else round(float(res[item]), 5)
            cards.append((f'HIERARCH {prefix} {name} {kw}', value, f'{label} {comment}'))
    return cards


def skyline_plane_columns(prefix: str = 'SKYPLANE') -> Dict[str, str]:
    """Returns the drpall column name of each header keyword of :func:`skyline_plane_cards`.

    Column names are lowercase, e.g. ``skyplane_z_gx`` for
    ``HIERARCH SKYPLANE Z GX`` and ``skyplane_skye_b_f1`` for
    ``HIERARCH SKYPLANE SKYE B F1``.

    Args:
        prefix: the first word of the keywords.

    Returns:
        A dict {column name: header keyword}, with the keyword as
        :func:`astropy.io.fits.Header.get` expects it (without ``HIERARCH``).
    """
    columns = {}
    for key, (name, _) in _HEADER_NAMES.items():
        items = _SKY_ITEMS if key.startswith('sky') else _PLANE_ITEMS
        for _, kw, _ in items:
            keyword = f'{prefix} {name} {kw}'
            columns[keyword.lower().replace(' ', '_')] = keyword
    return columns


def measure_cframe(rss, write_header: bool = True, raise_errors: bool = True,
                   **kwargs) -> Optional[Dict[str, Optional[Dict[str, float]]]]:
    """Measures the sky-line planes of a channel-combined frame (lvmCFrame).

    Args:
        rss: an :class:`lvmdrp.core.rss.lvmCFrame` (or any RSS with a 1D
            wavelength array and a slitmap).
        write_header: write the results to the header of `rss`
            (see :func:`skyline_plane_cards`).
        raise_errors: if False, a failed measurement is logged as a warning
            and None is returned, so that a reduction can continue.
        **kwargs: passed to :func:`measure_skyline_planes`.

    Returns:
        The dict returned by :func:`measure_skyline_planes`, or None if the
        measurement failed and `raise_errors` is False.
    """
    try:
        if rss._wave is None or np.ndim(rss._wave) != 1:
            raise ValueError('measure_cframe needs a frame with a 1D wavelength array')
        flux = rss._data
        mask = getattr(rss, '_mask', None)
        if mask is not None and np.shape(mask) == np.shape(flux) and np.any(mask):
            # masked pixels (cosmic rays, bad columns) are NaN in the written frame; leave out
            # the fibers they touch here too, so that the measurement is that of the product
            flux = np.where(mask, np.nan, flux)
        results = measure_skyline_planes(flux, rss._wave, rss._slitmap, **kwargs)
    except Exception as err:
        if raise_errors:
            raise
        log.warning(f'could not measure the sky-line planes: {err}')
        return None
    if write_header:
        for kw, value, comment in skyline_plane_cards(results):
            rss.setHdrValue(kw, value, comment)
    return results


# ----------------------------------------------------------------------------------------------
# The model of the large-scale flat-field error, fitted to the sky-line planes of many exposures
# ----------------------------------------------------------------------------------------------

# the components modelled per channel: spectrograph factors (%, relative to a mean of 1) and plane
MODEL_COMPONENTS = ('f1', 'f2', 'f3', 'gx', 'gy')
# terms shared by all epochs: focus (TESCIFO - focus0, its square, max(0, -focus)) and K-mirror
SHARED_TERMS = ('focus', 'focus2', 'focus_hinge', 'sinK', 'cosK', 'sin2K', 'cos2K')
# terms of each epoch: a constant and Legendre polynomials of degree 1-3 of the MJD within it
EPOCH_TERMS = ('c0', 't1', 't2', 't3')
# reference telescope focus (TESCIFO)
FOCUS0 = 35.5
# rejection of outlying exposures, in robust standard deviations from the epoch median
OUTLIER_SIGMA = 6.0
# exposures with fewer than this fraction of the usual number of fitted fibers are left out
MIN_FIBER_FRACTION = 0.85


def _shared_values(focus: np.ndarray, kmir: np.ndarray, focus0: float = FOCUS0) -> Dict[str, np.ndarray]:
    """Values of the shared terms for the given focus (TESCIFO) and K-mirror angle (deg)."""
    f = np.asarray(focus, dtype=float) - focus0
    k = np.radians(np.asarray(kmir, dtype=float))
    return {'focus': f, 'focus2': f * f, 'focus_hinge': np.maximum(0.0, -f),
            'sinK': np.sin(k), 'cosK': np.cos(k), 'sin2K': np.sin(2 * k), 'cos2K': np.cos(2 * k)}


def _epoch_values(mjd: np.ndarray, lo: float, hi: float) -> Dict[str, np.ndarray]:
    """Values of the epoch terms for MJDs in an epoch whose data span [lo, hi]."""
    u = np.clip(2 * (np.asarray(mjd, dtype=float) - lo) / max(1.0, hi - lo) - 1, -1, 1)
    return {'c0': np.ones_like(u), 't1': u, 't2': 0.5 * (3 * u ** 2 - 1),
            't3': 0.5 * (5 * u ** 3 - 3 * u)}


def assign_epochs(mjd: np.ndarray, epoch_mjds: Sequence[int]) -> np.ndarray:
    """Returns the epoch of each MJD: the latest epoch MJD not after it (the first epoch before all)."""
    starts = np.array(sorted(int(e) for e in epoch_mjds))
    i = np.clip(np.searchsorted(starts, np.asarray(mjd, dtype=float), side='right') - 1, 0, starts.size - 1)
    return starts[i]


def model_design(mjd: np.ndarray, focus: np.ndarray, kmir: np.ndarray, epoch: np.ndarray,
                 ranges: Dict[int, Tuple[float, float]],
                 focus0: float = FOCUS0,
                 epoch_terms: Sequence[str] = EPOCH_TERMS) -> Tuple[np.ndarray, list]:
    """Design matrix of the model: epoch constants and drifts, and the shared terms.

    Args:
        mjd: MJD of each exposure.
        focus: telescope focus (TESCIFO) of each exposure.
        kmir: K-mirror angle (TESCIKM, deg) of each exposure.
        epoch: epoch of each exposure (see :func:`assign_epochs`).
        ranges: the MJD range [lo, hi] of the data of each epoch.
        focus0: reference focus.
        epoch_terms: the epoch terms to include (a subset of `EPOCH_TERMS`).

    Returns:
        The design matrix (exposures x terms) and the term names, ``<term>_<epoch>``
        for the epoch terms and the names in `SHARED_TERMS`.
    """
    epoch = np.asarray(epoch)
    cols, names = [], []
    for e in sorted(ranges):
        sel = epoch == e
        vals = _epoch_values(mjd, *ranges[e])
        for term in epoch_terms:
            cols.append(np.where(sel, vals[term], 0.0))
            names.append(f'{term}_{e}')
    shared = _shared_values(focus, kmir, focus0)
    for term in SHARED_TERMS:
        cols.append(shared[term])
        names.append(term)
    return np.vstack(cols).T, names


def _column(table, name: str) -> np.ndarray:
    """A column of a drpall table as floats, NaN where missing or flagged (<= -999)."""
    if name not in table:
        return np.full(len(table), np.nan)
    v = np.asarray(table[name], dtype=float)
    return np.where(v <= -999, np.nan, v)


def model_targets(table, channel: str) -> np.ndarray:
    """The modelled components of each exposure, from the drpall sky-line plane columns.

    Args:
        table: drpall table (pandas DataFrame or astropy Table) with the ``skyplane_*`` columns.
        channel: 'b', 'r' or 'z'.

    Returns:
        An array (exposures x 5): the spectrograph factors in % relative to a
        mean of 1, and gx, gy in % per 10 mm (see `MODEL_COMPONENTS`).
    """
    f = np.column_stack([_column(table, f'skyplane_{channel}_f{k}') for k in (1, 2, 3)])
    with np.errstate(invalid='ignore'):
        f = 100 * (f / np.mean(f, axis=1)[:, None] - 1)
    return np.column_stack([f, _column(table, f'skyplane_{channel}_gx'),
                            _column(table, f'skyplane_{channel}_gy')])


def select_exposures(table, channel: str, epoch: np.ndarray, minexp: float = 300.0) -> np.ndarray:
    """Selects the exposures used to fit the model of one channel.

    Used: DRPQUAL 0, at least `minexp` seconds, focus and K-mirror angle known,
    all components measured, at least `MIN_FIBER_FRACTION` of the usual number of
    fibers fitted, and no component more than `OUTLIER_SIGMA` robust standard
    deviations from the median of its epoch.

    Args:
        table: drpall table with the ``skyplane_*`` columns.
        channel: 'b', 'r' or 'z'.
        epoch: epoch of each exposure.
        minexp: shortest exposure time used, s.

    Returns:
        True for the exposures to use.
    """
    V = model_targets(table, channel)
    nfib = _column(table, f'skyplane_{channel}_n')
    ok = np.all(np.isfinite(V), axis=1) & np.isfinite(nfib)
    ok &= np.isfinite(_column(table, 'sci_focpos')) & np.isfinite(_column(table, 'sci_kmpos'))
    ok &= _column(table, 'exptime') >= minexp
    if 'drpqual' in table:
        ok &= np.asarray(table['drpqual']) == 0
    if not ok.any():
        return ok
    ok &= nfib >= MIN_FIBER_FRACTION * np.median(nfib[ok])
    epoch = np.asarray(epoch)
    keep = ok.copy()
    for e in np.unique(epoch[ok]):
        sel = ok & (epoch == e)
        for j in range(V.shape[1]):
            v = V[sel, j]
            med = np.median(v)
            sig = 1.4826 * np.median(np.abs(v - med))
            if sig > 0:
                keep[np.flatnonzero(sel)[np.abs(v - med) > OUTLIER_SIGMA * sig]] = False
    return keep


def fit_flat_model(table, epoch_mjds: Sequence[int], channels: str = 'brz', minexp: float = 300.0,
                   focus0: float = FOCUS0, shared: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
                   epoch_terms: Sequence[str] = EPOCH_TERMS) -> Dict:
    """Fits the model of the large-scale flat-field error to the sky-line planes of many exposures.

    For each channel and component, the measured value is fitted as a constant
    and a cubic drift (Legendre polynomials in the MJD) within each epoch, plus
    focus and K-mirror terms shared by all epochs. The model describes what is
    left after the flats used in the reduction the table comes from.

    Args:
        table: drpall table (pandas DataFrame or astropy Table) with mjd,
            exptime, drpqual, sci_focpos, sci_kmpos and the ``skyplane_*`` columns.
        epoch_mjds: the first MJD of each epoch (the ffactor epochs).
        channels: the channels to fit.
        minexp: shortest exposure time used, s.
        focus0: reference focus.
        shared: the shared focus and K-mirror terms to hold fixed, {channel:
            {component: {term: value}}} (e.g. from :func:`read_shared_terms`);
            only the epoch terms are then fitted. The shared terms describe how
            the error changes with the state of the telescope, which doesn't
            depend on the flats, so they carry over from a fit to many more
            exposures; the epoch terms take up what the flats leave.
        epoch_terms: the epoch terms fitted (a subset of `EPOCH_TERMS`; e.g.
            ``('c0',)`` for a few nights per epoch); the others are zero.

    Returns:
        A dict with 'focus0', 'ranges' ({epoch: (lo, hi)}, the MJD range of the
        data of each epoch), 'nexp' ({channel: {epoch: number of exposures}}),
        'coeffs' ({channel: {component: {term: value}}}, terms as in
        :func:`model_design`) and 'shared_fixed' (whether the shared terms were held).

    Raises:
        ValueError: if a channel has fewer exposures than terms to fit.
    """
    mjd = _column(table, 'mjd')
    focus = _column(table, 'sci_focpos')
    kmir = _column(table, 'sci_kmpos')
    epoch = assign_epochs(mjd, epoch_mjds)
    usable = {ch: select_exposures(table, ch, epoch, minexp=minexp) for ch in channels}
    anyuse = np.any(np.column_stack(list(usable.values())), axis=1)
    ranges = {int(e): (float(mjd[anyuse & (epoch == e)].min()), float(mjd[anyuse & (epoch == e)].max()))
              for e in np.unique(epoch[anyuse])}
    model = {'focus0': focus0, 'ranges': ranges, 'nexp': {}, 'coeffs': {}, 'shared_fixed': shared is not None}
    for ch in channels:
        use = usable[ch] & np.isin(epoch, list(ranges))
        A, names = model_design(mjd[use], focus[use], kmir[use], epoch[use], ranges, focus0,
                                epoch_terms=epoch_terms)
        V = model_targets(table, ch)[use]
        if shared is not None:
            nshared = len(SHARED_TERMS)
            fixed = np.array([[shared[ch][comp][t] for comp in MODEL_COMPONENTS] for t in SHARED_TERMS])
            V = V - A[:, -nshared:] @ fixed
            A, names = A[:, :-nshared], names[:-nshared]
        if use.sum() <= A.shape[1]:
            raise ValueError(f'channel {ch}: {use.sum()} usable exposures for {A.shape[1]} terms')
        coef, *_ = np.linalg.lstsq(A, V, rcond=None)
        coeffs = {comp: dict(zip(names, map(float, coef[:, j]))) for j, comp in enumerate(MODEL_COMPONENTS)}
        for comp in MODEL_COMPONENTS:
            for e in ranges:
                for t in EPOCH_TERMS:
                    coeffs[comp].setdefault(f'{t}_{e}', 0.0)
            if shared is not None:
                coeffs[comp].update({t: float(shared[ch][comp][t]) for t in SHARED_TERMS})
        model['coeffs'][ch] = coeffs
        model['nexp'][ch] = {int(e): int(np.sum(epoch[use] == e)) for e in ranges}
    return model


def read_shared_terms(paths: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """The shared focus and K-mirror terms of existing model files, to hold fixed in :func:`fit_flat_model`.

    Args:
        paths: {channel: model file} (see :func:`write_flat_model`).

    Returns:
        {channel: {component: {term: value}}} for the terms in `SHARED_TERMS`.
    """
    out = {}
    for ch, p in paths.items():
        coeffs = read_flat_model(p)['coeffs']
        out[ch] = {comp: {t: coeffs[comp][t] for t in SHARED_TERMS} for comp in MODEL_COMPONENTS}
    return out


def model_id(epoch: int, channel: str, terms: Sequence[str], coeffs: Dict, source: Sequence[str]) -> str:
    """A short identifier of a model, from its epoch, channel and coefficients.

    Model files always have the same name, so a refit replaces a file without changing
    its name; the identifier, recorded in the model file and in every frame it
    corrects, tells the versions apart.

    Args:
        epoch: the epoch (its first MJD).
        channel: 'b', 'r' or 'z'.
        terms: the term names, in the order of the file.
        coeffs: {component: {source term: coefficient}}, as in :func:`fit_flat_model`.
        source: the key in `coeffs` of each term in `terms`.

    Returns:
        12 hexadecimal characters.
    """
    text = f'{int(epoch)} {channel} ' + ' '.join(
        f'{t}:' + ','.join(f'{coeffs[comp][s]:.10e}' for comp in MODEL_COMPONENTS)
        for t, s in zip(terms, source))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def write_flat_model(model: Dict, epoch: int, channel: str, out_path: str, drpver: str = '') -> None:
    """Writes the model of one epoch and channel as a calibration file (FITS).

    The file holds everything needed to evaluate the model for that epoch: the
    shared focus and K-mirror terms and the epoch's own constant and drift.

    Args:
        model: the dict returned by :func:`fit_flat_model`.
        epoch: the epoch (its first MJD).
        channel: 'b', 'r' or 'z'.
        out_path: the output file.
        drpver: version of the reduction the model was fitted to, recorded in the header.
    """
    from astropy.io import fits
    coeffs = model['coeffs'][channel]
    terms = list(EPOCH_TERMS) + list(SHARED_TERMS)
    source = [f'{t}_{epoch}' if t in EPOCH_TERMS else t for t in terms]
    cols = [fits.Column(name='term', format='12A', array=np.array(terms))]
    for comp in MODEL_COMPONENTS:
        cols.append(fits.Column(name=comp, format='D',
                                array=np.array([coeffs[comp][s] for s in source])))
    lo, hi = model['ranges'][epoch]
    hdr = fits.Header()
    hdr['CHANNEL'] = (channel, 'spectrograph channel')
    hdr['EPOCH'] = (int(epoch), 'first MJD of the model epoch')
    hdr['EPOCHLO'] = (lo, 'first MJD of the data fitted in this epoch')
    hdr['EPOCHHI'] = (hi, 'last MJD of the data fitted in this epoch')
    hdr['FOCUS0'] = (model['focus0'], 'reference telescope focus (TESCIFO)')
    hdr['NEXP'] = (model['nexp'][channel][epoch], 'exposures fitted in this epoch')
    hdr['NEXPALL'] = (sum(model['nexp'][channel].values()), 'exposures fitted in all epochs')
    hdr['DRPVER'] = (drpver, 'DRP version of the frames fitted')
    hdr['SHAREDFX'] = (bool(model.get('shared_fixed', False)), 'shared focus/K-mirror terms held fixed')
    if model.get('shared_from'):
        hdr['SHAREDFR'] = (str(model['shared_from'])[-68:], 'where the fixed shared terms come from')
    # how the model was made, so that it can be reproduced
    fit = model.get('provenance', {})
    if fit.get('drpall'):
        hdr['DRPALL'] = (str(fit['drpall'])[-68:], 'drpall file fitted')
    if fit.get('epoch_terms'):
        hdr['EPTERMS'] = (','.join(fit['epoch_terms']), 'epoch terms fitted')
    if fit.get('minexp') is not None:
        hdr['MINEXP'] = (float(fit['minexp']), 'shortest exposure time fitted [s]')
    hdr['FITDATE'] = (datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'), 'UTC date of the fit')
    hdr['LVMDRP'] = (get_lvmdrp_version(), 'lvmdrp commit of the fit')
    hdr['LVMCORE'] = (get_lvmcore_version(), 'lvmcore commit of the fit')
    hdr['MODELID'] = (model_id(epoch, channel, terms, coeffs, source),
                      'identifier of this model (hash of its coefficients)')
    hdr['COMMENT'] = ('Large-scale flat-field error left after the flats used (sdss/lvmdrp#268): '
                      'f1-f3 spectrograph factors in % (mean 1), gx, gy plane in % per 10 mm. '
                      'Value = sum over terms of coefficient x term value; see '
                      'lvmdrp.functions.flatgradient.evaluate_flat_model.')
    hdus = fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name='COEFFS')])
    hdus.writeto(out_path, overwrite=True)


def read_flat_model(path: str) -> Dict:
    """Reads a calibration file written by :func:`write_flat_model`.

    Args:
        path: the file.

    Returns:
        A dict with 'channel', 'epoch', 'range' (lo, hi), 'focus0' and 'coeffs'
        ({component: {term: value}}, terms as in `EPOCH_TERMS` and `SHARED_TERMS`).
    """
    from astropy.io import fits
    with fits.open(path) as hdus:
        hdr = hdus[0].header
        tab = hdus['COEFFS'].data
        terms = [t.strip() for t in tab['term']]
        coeffs = {comp: dict(zip(terms, map(float, tab[comp]))) for comp in MODEL_COMPONENTS}
    return {'channel': hdr['CHANNEL'], 'epoch': int(hdr['EPOCH']), 'range': (hdr['EPOCHLO'], hdr['EPOCHHI']),
            'focus0': float(hdr['FOCUS0']), 'coeffs': coeffs, 'modelid': str(hdr.get('MODELID', ''))}


def evaluate_flat_model(model: Dict, mjd: float, focus: float, kmir: float) -> Dict[str, float]:
    """Evaluates a model read by :func:`read_flat_model` for one exposure.

    The drift is held at its end values outside the MJD range of the data fitted.

    Args:
        model: the dict returned by :func:`read_flat_model`.
        mjd: MJD of the exposure.
        focus: telescope focus (TESCIFO).
        kmir: K-mirror angle (TESCIKM, deg).

    Returns:
        {component: value}: the factors in % (relative to a mean of 1) and gx, gy
        in % per 10 mm.
    """
    vals = dict(_epoch_values(np.atleast_1d(mjd), *model['range']))
    vals.update(_shared_values(np.atleast_1d(focus), np.atleast_1d(kmir), model['focus0']))
    return {comp: float(sum(c * vals[t][0] for t, c in model['coeffs'][comp].items()))
            for comp in MODEL_COMPONENTS}


def read_drpall(drpall: str):
    """Reads a drpall summary file (HDF5 or FITS) into a pandas DataFrame."""
    import pandas as pd
    if drpall.endswith('.h5'):
        return pd.read_hdf(drpall, key='summary')
    from astropy.table import Table
    tab = Table.read(drpall)
    return tab[[c for c in tab.colnames if len(tab[c].shape) <= 1]].to_pandas()


def create_flat_models(drpall: str, epoch_mjds: Sequence[int], channels: str = 'brz', drpver: str = '',
                       out_dir: Optional[str] = None, from_sandbox: bool = True,
                       minexp: float = 300.0, dry_run: bool = False, shared_from: Optional[str] = None,
                       epoch_terms: Sequence[str] = EPOCH_TERMS) -> Dict[Tuple[int, str], str]:
    """Fits the flat-field error model to a drpall file and writes one calibration file per epoch and channel.

    Args:
        drpall: the drpall file (HDF5 or FITS) of the reduction to fit, with
            the ``skyplane_*`` columns.
        epoch_mjds: the first MJD of each epoch (the ffactor epochs).
        channels: the channels to fit.
        drpver: DRP version of the reduction, recorded in the files and used to
            find the calibration directory when `from_sandbox` is False.
        out_dir: write ``<out_dir>/<epoch>/lvm-mfiberflat_gradient-<channel>.fits`` instead
            of into the calibration directories.
        from_sandbox: write into the sandbox calibration directories
            ($LVM_MASTER_DIR) rather than those of the `drpver` reduction.
        minexp: shortest exposure time used, s.
        dry_run: fit and log the result, but write nothing.
        shared_from: a directory with ``lvm-mfiberflat_gradient-<channel>.fits`` files
            whose shared focus and K-mirror terms are held fixed (see :func:`fit_flat_model`).
        epoch_terms: the epoch terms fitted.

    Returns:
        {(epoch, channel): path} of the files written (or that would be written).
    """
    import os
    from lvmdrp.utils.paths import get_calib_paths

    table = read_drpall(drpall)
    log.info(f'fitting the flat-field error model to {len(table)} exposures in {drpall}')
    shared = None
    if shared_from is not None:
        files = {ch: os.path.join(shared_from, f'lvm-mfiberflat_gradient-{ch}.fits') for ch in channels}
        missing = [f for f in files.values() if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f'model files for the shared terms not found: {missing}')
        shared = read_shared_terms(files)
        log.info(f'holding the shared focus and K-mirror terms fixed at those of {shared_from}')
    model = fit_flat_model(table, epoch_mjds, channels=channels, minexp=minexp, shared=shared,
                           epoch_terms=epoch_terms)
    model['shared_from'] = shared_from
    model['provenance'] = {'drpall': os.path.abspath(drpall), 'epoch_terms': list(epoch_terms), 'minexp': minexp}
    paths = {}
    for epoch in sorted(model['ranges']):
        if out_dir is not None:
            out = {ch: os.path.join(out_dir, str(epoch), f'lvm-mfiberflat_gradient-{ch}.fits') for ch in channels}
        else:
            calibs = get_calib_paths(mjd=epoch, version=drpver or None, cameras='*',
                                     flavors={'fiberflat_gradient'}, from_sandbox=from_sandbox)
            out = calibs['fiberflat_gradient']
        for ch in channels:
            paths[(epoch, ch)] = out[ch]
            c = model['coeffs'][ch]
            log.info(f'epoch {epoch} channel {ch}: {model["nexp"][ch][epoch]} exposures, '
                     f'constant gx {c["gx"][f"c0_{epoch}"]:.2f}, gy {c["gy"][f"c0_{epoch}"]:.2f} %/10mm '
                     f'-> {out[ch]}')
            if not dry_run:
                os.makedirs(os.path.dirname(out[ch]), exist_ok=True)
                write_flat_model(model, epoch, ch, out[ch], drpver=drpver)
    return paths


# ----------------------------------------------------------------------------------------------
# Applying the model (and optionally a direct fit of what is left) to a channel-combined frame
# ----------------------------------------------------------------------------------------------

# largest difference between the 5577 and 8399 residual planes for the direct step, %/10 mm
DIRECT_GUARD = 1.5
# largest difference between the r OH residual plane and the b, z mean for r OH to be used, %/10 mm
DIRECT_RGUARD = 1.5
# keyword prefix of the record of the correction in the header
APPLIED_PREFIX = 'FLATGRAD'


def channel_weights(wave: np.ndarray) -> Dict[str, np.ndarray]:
    """Weight of each channel at each wavelength of a channel-combined frame.

    1 inside a channel, with linear ramps across the b-r and r-z overlaps
    (`lvmdrp.core.constants.SPEC_CHANNELS`).

    Args:
        wave: wavelength of each pixel, Angstrom.

    Returns:
        {channel: weight of each pixel}; the weights add up to 1 at every pixel.
    """
    from lvmdrp.core.constants import SPEC_CHANNELS
    wave = np.asarray(wave, dtype=float)
    w = {ch: np.zeros(wave.size) for ch in 'brz'}
    br0, br1 = SPEC_CHANNELS['r'][0], SPEC_CHANNELS['b'][1]
    rz0, rz1 = SPEC_CHANNELS['z'][0], SPEC_CHANNELS['r'][1]
    w['b'][wave < br0] = 1.0
    m = (wave >= br0) & (wave <= br1)
    w['r'][m] = (wave[m] - br0) / (br1 - br0)
    w['b'][m] = 1.0 - w['r'][m]
    w['r'][(wave > br1) & (wave < rz0)] = 1.0
    m = (wave >= rz0) & (wave <= rz1)
    w['z'][m] = (wave[m] - rz0) / (rz1 - rz0)
    w['r'][m] = 1.0 - w['z'][m]
    w['z'][wave > rz1] = 1.0
    return w


def fiber_throughput(factors: np.ndarray, gx: float, gy: float, x: np.ndarray, y: np.ndarray,
                     spec: np.ndarray) -> np.ndarray:
    """Relative throughput of each fiber for spectrograph factors and a plane, normalized to a mean of 1.

    Args:
        factors: the three spectrograph factors (fractions, e.g. 1.01).
        gx: x slope of the plane, % per 10 mm.
        gy: y slope of the plane, % per 10 mm.
        x: fiber x positions, mm.
        y: fiber y positions, mm.
        spec: spectrograph of each fiber (1, 2 or 3).

    Returns:
        The throughput of each fiber.
    """
    k = np.asarray(factors, dtype=float)[np.asarray(spec, dtype=int) - 1] * (1 + (gx * x + gy * y) / 1000)
    return k / np.mean(k)


def _residual_planes(flux: np.ndarray, wave: np.ndarray, x: np.ndarray, y: np.ndarray,
                     spec: np.ndarray) -> Dict[str, Optional[Dict[str, float]]]:
    """The planes and factors left in corrected science spectra: 5577 (b), 8399 (z), r OH sum (r)."""
    out = {}
    for ch in 'bzr':
        stars = star_mask(flux, wave, ch)
        if ch == 'r':
            z = np.sum([line_flux(flux, wave, HELDOUT_LINES[m][1], HELDOUT_LINES[m][2])
                        for m in HELDOUT_GROUPS['r_oh'][1]], axis=0)
        else:
            cwave, sides = SKYLINES[ch]
            z = line_flux(flux, wave, (cwave - LINE_HALF_WIDTH, cwave + LINE_HALF_WIDTH), sides)
        out[ch] = fit_plane(x, y, np.where(stars, np.nan, z), spec)
    return out


def apply_flat_models(rss, models: Dict[str, Dict], direct: bool = False, guard: float = DIRECT_GUARD,
                      rguard: float = DIRECT_RGUARD,
                      model_files: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Corrects the science fibers of a channel-combined frame for the large-scale flat-field error.

    The model of each channel (see :func:`read_flat_model`) is evaluated from the
    header (MJD, TESCIFO, TESCIKM), and the flux and error of the science fibers are
    divided by its spectrograph factors times its plane (normalized to a mean of 1
    over the science fibers), blended across the channel overlaps. With `direct`,
    the planes left in 5577 (b), 8399 (z) and the r OH lines are then fitted and
    divided out as well, if the b and z planes agree to within `guard`; r gets the
    average of its r OH plane and the b, z mean if they agree to within `rguard`,
    otherwise the b, z mean. What was applied is recorded in the header
    (``HIERARCH FLATGRAD ...``); a frame already corrected is left as it is.

    Args:
        rss: an :class:`lvmdrp.core.rss.lvmCFrame` (1D wavelength array), changed in place.
        models: {channel: model} for the channels b, r and z.
        direct: also apply the direct fit of what is left.
        guard: largest 5577 - 8399 residual plane difference for the direct step, %/10 mm.
        rguard: largest r OH - (b, z mean) difference for r OH to be used, %/10 mm.
        model_files: {channel: file name}, recorded in the header.

    Returns:
        A dict with the model values ('model': {channel: {component: value}}),
        'direct' (whether the direct step was applied), 'guard', 'rmode' and the
        residual planes ('residual'); None if the frame was not corrected.
    """
    hdr = rss._header
    if hdr.get(f'{APPLIED_PREFIX} APPLIED', False):
        log.warning('the large-scale flat-field correction was already applied; frame left as it is')
        return None
    missing = [ch for ch in 'brz' if ch not in models]
    if missing:
        log.warning(f'no flat-field error model for channel(s) {missing}; frame not corrected')
        return None
    mjd, focus, kmir = (hdr.get(k) for k in ('MJD', 'TESCIFO', 'TESCIKM'))
    if any(v is None or not np.isfinite(float(v)) or float(v) <= -900 for v in (mjd, focus, kmir)):
        log.warning(f'MJD, TESCIFO or TESCIKM missing ({mjd}, {focus}, {kmir}); frame not corrected')
        return None
    if rss._wave is None or np.ndim(rss._wave) != 1:
        log.warning('the frame has no 1D wavelength array; frame not corrected')
        return None

    slitmap = rss._slitmap
    sci = np.flatnonzero(np.asarray(slitmap['telescope']).astype(str) == 'Sci')
    x = np.asarray(slitmap['xpmm'], dtype=float)[sci]
    y = np.asarray(slitmap['ypmm'], dtype=float)[sci]
    spec = np.asarray(slitmap['spectrographid'], dtype=int)[sci]
    wave = np.asarray(rss._wave, dtype=float)
    weights = channel_weights(wave)

    values = {ch: evaluate_flat_model(models[ch], float(mjd), float(focus), float(kmir)) for ch in 'brz'}
    corr = np.zeros((sci.size, wave.size))
    for ch in 'brz':
        v = values[ch]
        f = 1 + np.array([v['f1'], v['f2'], v['f3']]) / 100
        corr += fiber_throughput(f / f.mean(), v['gx'], v['gy'], x, y, spec)[:, None] * weights[ch][None, :]

    result = {'model': values, 'direct': False, 'guard': np.nan, 'rguard': np.nan, 'rmode': 'none',
              'residual': None}
    if direct:
        res = _residual_planes(rss._data[sci] / corr, wave, x, y, spec)
        result['residual'] = res
        nmin = 0.7 * MIN_FIBER_FRACTION * sci.size
        if res['b'] is not None and res['z'] is not None:
            result['guard'] = float(np.hypot(res['b']['gx'] - res['z']['gx'], res['b']['gy'] - res['z']['gy']))
            if result['guard'] <= guard and min(res['b']['n'], res['z']['n']) >= nmin:
                keys = ('f1', 'f2', 'f3', 'gx', 'gy')
                rfit = {ch: res[ch] for ch in 'bz'}
                rfit['r'] = {k: 0.5 * (res['b'][k] + res['z'][k]) for k in keys}
                result['rmode'] = 'bz'
                if res['r'] is not None:
                    result['rguard'] = float(np.hypot(res['r']['gx'] - rfit['r']['gx'],
                                                      res['r']['gy'] - rfit['r']['gy']))
                    if result['rguard'] <= rguard and res['r']['n'] >= nmin:
                        rfit['r'] = {k: 0.5 * (res['r'][k] + rfit['r'][k]) for k in keys}
                        result['rmode'] = 'oh+bz'
                cres = np.zeros_like(corr)
                for ch in 'brz':
                    r = rfit[ch]
                    k = fiber_throughput(np.array([r['f1'], r['f2'], r['f3']]), r['gx'], r['gy'], x, y, spec)
                    cres += k[:, None] * weights[ch][None, :]
                corr = corr * cres
                result['direct'] = True

    rss._data[sci] = rss._data[sci] / corr
    if rss._error is not None:
        rss._error[sci] = rss._error[sci] / corr

    p = APPLIED_PREFIX
    rss.setHdrValue(f'HIERARCH {p} APPLIED', True, 'large-scale flat-field correction applied (#268)')
    epochs = sorted({int(models[ch]['epoch']) for ch in 'brz'})
    rss.setHdrValue(f'HIERARCH {p} EPOCH', epochs[0], 'ffactor epoch of the model')
    for ch in 'brz':
        name = (model_files or {}).get(ch)
        if name:
            import os
            rss.setHdrValue(f'HIERARCH {p} {ch.upper()} FILE', os.path.basename(name), 'model file')
        if models[ch].get('modelid'):
            rss.setHdrValue(f'HIERARCH {p} {ch.upper()} MODELID', models[ch]['modelid'], 'model identifier')
        for comp in MODEL_COMPONENTS:
            unit = '%' if comp.startswith('f') else '%/10mm'
            rss.setHdrValue(f'HIERARCH {p} {ch.upper()} {comp.upper()}', round(values[ch][comp], 4),
                            f'{ch} model {comp} [{unit}]')
    rss.setHdrValue(f'HIERARCH {p} DIRECT', result['direct'], 'direct fit of what is left applied')
    if direct:
        rss.setHdrValue(f'HIERARCH {p} DGUARD', round(result['guard'], 3) if np.isfinite(result['guard']) else -1.0,
                        '|5577 - 8399| residual plane [%/10mm]')
        rss.setHdrValue(f'HIERARCH {p} DRGUARD',
                        round(result['rguard'], 3) if np.isfinite(result['rguard']) else -1.0,
                        '|r OH - mean(b, z)| residual plane [%/10mm]')
        rss.setHdrValue(f'HIERARCH {p} DRMODE', result['rmode'], 'r direct step: oh+bz, bz or none')
        for ch, label in (('b', '5577'), ('z', '8399'), ('r', 'r OH')):
            r = result['residual'][ch]
            for comp in ('gx', 'gy'):
                rss.setHdrValue(f'HIERARCH {p} D{ch.upper()} {comp.upper()}',
                                round(r[comp], 3) if r else -999.0, f'{label} residual plane {comp} [%/10mm]')
    log.info('large-scale flat-field correction applied: model planes ' +
             ', '.join(f'{ch} ({values[ch]["gx"]:.2f}, {values[ch]["gy"]:.2f})' for ch in 'brz') +
             f' %/10mm; direct step {"applied" if result["direct"] else "not applied"}' +
             (f' (guard {result["guard"]:.2f}, r {result["rmode"]})' if direct else ''))
    return result


def find_flat_models(mjd: int, drpver: Optional[str] = None, from_sandbox: bool = True,
                     ffactor_epochs: Optional[Dict] = None, epochs_kind: str = 'ffactor') -> Dict[str, str]:
    """Finds the model files of the epoch of an MJD.

    Args:
        mjd: MJD of the exposure.
        drpver: DRP version, for the calibration directories of a reduction.
        from_sandbox: look in the sandbox calibration directories ($LVM_MASTER_DIR).
        ffactor_epochs: the ffactor epochs; default: those of lvmcore.
        epochs_kind: 'ffactor' (the model of the ffactor epoch, for flats with the
            stage-2 factors) or 'calibration' (the model in the calibration
            directory of the exposure's calibration epoch, next to its twilight flats).

    Returns:
        {channel: path} of the model files that exist (empty if none).
    """
    import os
    from lvmdrp.utils.paths import get_calib_paths
    if epochs_kind == 'calibration':
        calibs = get_calib_paths(mjd=mjd, version=drpver, flavors={'fiberflat_gradient'}, from_sandbox=from_sandbox)
        return {ch: p for ch, p in calibs.get('fiberflat_gradient', {}).items() if os.path.exists(p)}
    if epochs_kind != 'ffactor':
        raise ValueError(f"epochs_kind must be 'ffactor' or 'calibration', not {epochs_kind!r}")
    if ffactor_epochs is None:
        from lvmdrp.functions.run_calseq import load_epochs_file
        ffactor_epochs = load_epochs_file(epochs_kind='ffactor', verbose=False) or {}
    if not ffactor_epochs:
        return {}
    calibs = get_calib_paths(mjd=mjd, version=drpver, flavors={'fiberflat_gradient'}, epochs=ffactor_epochs,
                             from_sandbox=from_sandbox)
    return {ch: p for ch, p in calibs.get('fiberflat_gradient', {}).items() if os.path.exists(p)}


def correct_cframe(rss, mjd: int, drpver: Optional[str] = None, from_sandbox: bool = True,
                   direct: bool = False) -> Optional[Dict]:
    """Corrects a channel-combined frame with the model files of its epoch, if they exist.

    Never raises: a missing model or a failure is logged and the frame is left as it is.

    Args:
        rss: an :class:`lvmdrp.core.rss.lvmCFrame`, changed in place.
        mjd: MJD of the exposure, to find its ffactor epoch.
        drpver: DRP version, for the calibration directories of a reduction.
        from_sandbox: look in the sandbox calibration directories ($LVM_MASTER_DIR).
        direct: also apply the direct fit of what is left.

    Returns:
        The dict returned by :func:`apply_flat_models`, or None if the frame was not corrected.
    """
    try:
        files = find_flat_models(mjd, drpver=drpver, from_sandbox=from_sandbox)
        if len(files) < 3:
            log.info(f'no large-scale flat-field error model for MJD {mjd} (found {sorted(files)}); '
                     'frame not corrected')
            return None
        models = {ch: read_flat_model(p) for ch, p in files.items()}
        return apply_flat_models(rss, models, direct=direct, model_files=files)
    except Exception as err:
        log.warning(f'could not apply the large-scale flat-field correction: {err}')
        return None


# ----------------------------------------------------------------------------------------------
# Correcting the lvmFrames: all fibers, per channel, before the sky and the flux calibration
# ----------------------------------------------------------------------------------------------

# keyword prefix of the record of the correction of an lvmFrame
FRAME_PREFIX = 'FFCORR'


def correction_columns() -> Dict[str, Tuple[str, str]]:
    """The drpall columns of the large-scale flat-field corrections recorded in the frame headers.

    Both records are included: ``FFCORR`` (the correction of the lvmFrames, all
    fibers, see :func:`apply_frame_correction`) and ``FLATGRAD`` (the correction of
    the science fibers of the lvmCFrame, see :func:`apply_flat_models`). Column
    names are lowercase, e.g. ``ffcorr_z_gx`` for ``HIERARCH FFCORR Z GX`` and
    ``flatgrad_db_gx`` for ``HIERARCH FLATGRAD DB GX``. Flags (APPLIED, DIRECT)
    become 1 or 0.

    Returns:
        {column name: (header keyword without ``HIERARCH``, dtype)}, dtype 'float64' or 'str'.
    """
    cols = {}

    def add(prefix, kw, dtype='float64'):
        keyword = f'{prefix} {kw}'
        cols[keyword.lower().replace(' ', '_')] = (keyword, dtype)

    for prefix in (FRAME_PREFIX, APPLIED_PREFIX):
        add(prefix, 'APPLIED')
        add(prefix, 'EPOCH')
        for ch in 'BRZ':
            for comp in MODEL_COMPONENTS:
                add(prefix, f'{ch} {comp.upper()}')
    for prefix in (FRAME_PREFIX, APPLIED_PREFIX):
        for ch in 'BRZ':
            add(prefix, f'{ch} FILE', 'str')
            add(prefix, f'{ch} MODELID', 'str')
    add(FRAME_PREFIX, 'SOURCE', 'str')
    add(FRAME_PREFIX, 'RMODE', 'str')
    add(FRAME_PREFIX, 'GUARD')
    add(FRAME_PREFIX, 'RGUARD')
    add(APPLIED_PREFIX, 'DIRECT')
    add(APPLIED_PREFIX, 'DGUARD')
    add(APPLIED_PREFIX, 'DRGUARD')
    add(APPLIED_PREFIX, 'DRMODE', 'str')
    for ch in 'BZR':
        for comp in ('GX', 'GY'):
            add(APPLIED_PREFIX, f'D{ch} {comp}')
    return cols


def frame_correction(slitmap, coeffs: Dict[str, float]) -> np.ndarray:
    """The divisor of each fiber of one channel for the large-scale flat-field correction.

    The spectrograph factors apply to every fiber (science, sky and standard); the
    plane only to the science fibers, normalized to a mean of 1 over the good ones,
    so that the plane doesn't change the level of the science IFU relative to the
    other fibers.

    Args:
        slitmap: the slitmap table, with columns telescope, spectrographid, xpmm,
            ypmm and (optionally) fibstatus.
        coeffs: f1, f2, f3 (% relative to a mean of 1) and gx, gy (% per 10 mm),
            as in `MODEL_COMPONENTS`.

    Returns:
        The divisor of each fiber.
    """
    spec = np.asarray(slitmap['spectrographid'], dtype=int)
    sci = np.asarray(slitmap['telescope']).astype(str) == 'Sci'
    f = 1 + np.array([coeffs['f1'], coeffs['f2'], coeffs['f3']], dtype=float) / 100
    corr = (f / f.mean())[spec - 1]
    x = np.asarray(slitmap['xpmm'], dtype=float)
    y = np.asarray(slitmap['ypmm'], dtype=float)
    plane = 1 + (coeffs['gx'] * x + coeffs['gy'] * y) / 1000
    good = sci & np.isfinite(plane)
    if 'fibstatus' in slitmap.colnames:
        good &= np.asarray(slitmap['fibstatus'], dtype=int) == 0
    corr[sci] *= plane[sci] / np.mean(plane[good])
    return corr


def read_frame_correction(header, channel: str) -> Optional[Dict[str, float]]:
    """The correction recorded in an lvmFrame header by :func:`apply_frame_correction`.

    Args:
        header: the header of the lvmFrame.
        channel: 'b', 'r' or 'z'.

    Returns:
        {component: value} as in `MODEL_COMPONENTS`, or None if no correction is applied.
    """
    if not header.get(f'{FRAME_PREFIX} APPLIED', False):
        return None
    try:
        return {comp: float(header[f'{FRAME_PREFIX} {channel.upper()} {comp.upper()}'])
                for comp in MODEL_COMPONENTS}
    except KeyError:
        return None


def apply_frame_correction(frame, channel: str, coeffs: Optional[Dict[str, float]],
                           source: str = '', info: Optional[Dict] = None) -> np.ndarray:
    """Applies the large-scale flat-field correction to one channel's lvmFrame, replacing any earlier one.

    A correction already recorded in the header is divided out first, so the frame
    can be corrected again (or, with `coeffs` None, returned to the flat alone). The
    flux and error are divided by :func:`frame_correction`, and the SUPERFLAT
    extension is multiplied by it, so that it remains the flat applied.

    Args:
        frame: an :class:`lvmdrp.core.rss.lvmFrame`, changed in place.
        channel: 'b', 'r' or 'z'.
        coeffs: f1, f2, f3 (%) and gx, gy (% per 10 mm); None to remove the correction.
        source: how the coefficients were obtained (e.g. 'direct', 'model'), recorded.
        info: other values to record, {keyword part: (value, comment)}.

    Returns:
        The factor by which each fiber was divided (1 where nothing changed).
    """
    hdr = frame._header
    slitmap = frame._slitmap
    total = np.ones(len(slitmap))
    old = read_frame_correction(hdr, channel)
    if old is not None:
        total /= frame_correction(slitmap, old)
    if coeffs is not None:
        total *= frame_correction(slitmap, coeffs)

    dtype = frame._data.dtype
    frame._data = (frame._data / total[:, None]).astype(dtype)
    if frame._error is not None:
        frame._error = (frame._error / total[:, None]).astype(frame._error.dtype)
    superflat = getattr(frame, '_superflat', None)
    if superflat is not None:
        frame._superflat = (superflat * total[:, None]).astype(superflat.dtype)

    p, ch = FRAME_PREFIX, channel.upper()
    frame.setHdrValue(f'HIERARCH {p} APPLIED', coeffs is not None,
                      'large-scale flat-field correction applied (#268)')
    if coeffs is None:
        for comp in MODEL_COMPONENTS:
            hdr.remove(f'HIERARCH {p} {ch} {comp.upper()}', ignore_missing=True)
        log.info(f'large-scale flat-field correction of channel {channel} removed')
        return total
    frame.setHdrValue(f'HIERARCH {p} SOURCE', source, 'how the correction was obtained')
    for comp in MODEL_COMPONENTS:
        unit = '% (mean 1), all fibers' if comp.startswith('f') else '%/10mm, Sci fibers'
        frame.setHdrValue(f'HIERARCH {p} {ch} {comp.upper()}', round(float(coeffs[comp]), 5),
                          f'{channel} correction {comp} [{unit}]')
    for key, (value, comment) in (info or {}).items():
        frame.setHdrValue(f'HIERARCH {p} {key}', value, comment)
    log.info(f'large-scale flat-field correction of channel {channel} ({source}): factors '
             f'{coeffs["f1"]:.2f} {coeffs["f2"]:.2f} {coeffs["f3"]:.2f} %, plane '
             f'{coeffs["gx"]:.2f} {coeffs["gy"]:.2f} %/10mm')
    return total


def _measured_plane(header, name: str) -> Optional[Dict[str, float]]:
    """A sky-line plane from the SKYPLANE keywords of a header, factors in % (mean 1); None if missing."""
    try:
        f = np.array([float(header[f'SKYPLANE {name} F{k}']) for k in (1, 2, 3)])
        gx, gy = float(header[f'SKYPLANE {name} GX']), float(header[f'SKYPLANE {name} GY'])
        n = int(header[f'SKYPLANE {name} N'])
    except (KeyError, TypeError, ValueError):
        return None
    vals = np.append(f, [gx, gy])
    if n <= 0 or not np.all(np.isfinite(vals)) or np.any(vals <= -999) or np.any(f <= 0):
        return None
    f = 100 * (f / f.mean() - 1)
    return {'f1': f[0], 'f2': f[1], 'f3': f[2], 'gx': gx, 'gy': gy, 'n': n}


def _combine(first: Dict[str, float], second: Dict[str, float]) -> Dict[str, float]:
    """Two corrections in sequence: the factors multiplied (renormalized to a mean of 1), the planes added."""
    f = np.array([(1 + first[k] / 100) * (1 + second[k] / 100) for k in ('f1', 'f2', 'f3')])
    f = 100 * (f / f.mean() - 1)
    return {'f1': f[0], 'f2': f[1], 'f3': f[2], 'gx': first['gx'] + second['gx'],
            'gy': first['gy'] + second['gy']}


def _difference(total: Dict[str, float], part: Dict[str, float]) -> Dict[str, float]:
    """What is left of `total` after `part`: the inverse of :func:`_combine`."""
    f = np.array([(1 + total[k] / 100) / (1 + part[k] / 100) for k in ('f1', 'f2', 'f3')])
    f = 100 * (f / f.mean() - 1)
    return {'f1': f[0], 'f2': f[1], 'f3': f[2], 'gx': total['gx'] - part['gx'], 'gy': total['gy'] - part['gy']}


def frame_coefficients(header, models: Optional[Dict[str, Dict]] = None,
                       previous: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
                       direct: bool = True, guard: float = DIRECT_GUARD, rguard: float = DIRECT_RGUARD,
                       min_fibers: int = 0) -> Optional[Dict]:
    """The large-scale flat-field correction of each channel of one exposure, for its lvmFrames.

    The sky-line planes measured in the exposure's lvmCFrame (the ``SKYPLANE``
    keywords of :func:`measure_cframe`) are what the lvmFrames' correction at the
    time (`previous`, none for frames with the flat alone) left, so the total
    measured error of a channel is the previous correction combined with them. The
    lines used are 5577 (b), 8399 (z) and the summed r OH lines (r).

    - With `models` and `direct`: the residuals are the measured errors minus the
      models (evaluated from MJD, TESCIFO and TESCIKM). If the b and z residual
      planes agree to within `guard`, b and z take their measured errors, and r the
      model plus the mean of the b and z residuals, averaged with its own r OH
      residual if that agrees to within `rguard`. Otherwise the models are used.
    - With `models` alone: the models.
    - With `direct` and no models (a new epoch): b and z take their measured
      errors; r its r OH factors (6364 if r OH is missing) and the mean of the b
      and z planes, averaged with the r OH plane if it agrees to within `rguard`.
      There is no guard.

    Args:
        header: the lvmCFrame header, with the SKYPLANE keywords, MJD, TESCIFO and TESCIKM.
        models: {channel: model} as read by :func:`read_flat_model`, or None.
        previous: {channel: correction applied to the lvmFrames} (see
            :func:`read_frame_correction`), or None.
        direct: use the exposure's own measurements.
        guard: largest b - z residual plane difference for the direct step, %/10 mm.
        rguard: largest r OH - (b, z mean) plane difference for r OH to be used, %/10 mm.
        min_fibers: fewest fibers in the b and z plane fits for the direct step.

    Returns:
        A dict with 'coeffs' ({channel: {component: value}}), 'mode' ('direct' or
        'model'), 'rmode' ('oh+bz', 'bz', 'oh', '6364' or 'none'), 'guard' and
        'rguard' (NaN where not computed), 'model' ({channel: values} or None) and
        'measured' ({line: total error} or None for each of b, z, r_oh, r); None if
        no correction is possible.
    """
    previous = previous or {}
    names = {'b': 'B', 'z': 'Z', 'r_oh': 'ROH', 'r': 'R'}
    measured = {}
    for key, name in names.items():
        m = _measured_plane(header, name)
        prev = previous.get(key[0])
        measured[key] = _combine(prev, m) if (m is not None and prev is not None) else m
        if m is not None:
            measured[key]['n'] = m['n']
    out = {'mode': 'model', 'rmode': 'none', 'guard': np.nan, 'rguard': np.nan, 'model': None,
           'measured': measured}

    values = None
    if models is not None:
        mjd, focus, kmir = (header.get(k) for k in ('MJD', 'TESCIFO', 'TESCIKM'))
        if any(v is None or not np.isfinite(float(v)) or float(v) <= -900 for v in (mjd, focus, kmir)):
            log.warning(f'MJD, TESCIFO or TESCIKM missing ({mjd}, {focus}, {kmir}); model not used')
        elif any(ch not in models for ch in 'brz'):
            log.warning(f'no flat-field error model for some channels ({sorted(models)}); model not used')
        else:
            values = {ch: evaluate_flat_model(models[ch], float(mjd), float(focus), float(kmir)) for ch in 'brz'}
    out['model'] = values

    b, z, roh = measured['b'], measured['z'], measured['r_oh']
    have_bz = b is not None and z is not None and min(b['n'], z['n']) >= min_fibers
    keys = ('f1', 'f2', 'f3', 'gx', 'gy')
    if direct and have_bz and values is not None:
        res = {ch: _difference(measured[key], values[ch]) for ch, key in (('b', 'b'), ('z', 'z'))}
        out['guard'] = float(np.hypot(res['b']['gx'] - res['z']['gx'], res['b']['gy'] - res['z']['gy']))
        if out['guard'] <= guard:
            rres = {k: 0.5 * (res['b'][k] + res['z'][k]) for k in keys}
            out['rmode'] = 'bz'
            if roh is not None and roh['n'] >= min_fibers:
                own = _difference(roh, values['r'])
                out['rguard'] = float(np.hypot(own['gx'] - rres['gx'], own['gy'] - rres['gy']))
                if out['rguard'] <= rguard:
                    rres = {k: 0.5 * (own[k] + rres[k]) for k in keys}
                    out['rmode'] = 'oh+bz'
            out['coeffs'] = {'b': {k: b[k] for k in keys}, 'z': {k: z[k] for k in keys},
                             'r': _combine(values['r'], rres)}
            out['mode'] = 'direct'
            return out
    if values is not None:
        out['coeffs'] = {ch: dict(values[ch]) for ch in 'brz'}
        return out
    if not direct or not have_bz:
        log.warning('no flat-field error model, and ' +
                    ('the direct step not asked for' if not direct else 'no b and z sky-line planes') +
                    '; no correction')
        return None
    rown = roh if roh is not None else measured['r']
    if rown is None:
        log.warning('no r sky-line plane (r OH or 6364); no correction')
        return None
    plane = {k: 0.5 * (b[k] + z[k]) for k in ('gx', 'gy')}
    out['rmode'] = 'bz'
    out['rguard'] = float(np.hypot(rown['gx'] - plane['gx'], rown['gy'] - plane['gy']))
    if out['rguard'] <= rguard:
        plane = {k: 0.5 * (rown[k] + plane[k]) for k in ('gx', 'gy')}
        out['rmode'] = 'oh+bz' if roh is not None else '6364+bz'
    out['coeffs'] = {'b': {k: b[k] for k in keys}, 'z': {k: z[k] for k in keys},
                     'r': {**{k: rown[k] for k in ('f1', 'f2', 'f3')}, **plane}}
    out['mode'] = 'direct'
    log.info('no flat-field error model: direct step from the exposure alone, without the guard')
    return out


def correct_frames(frame_paths: Dict[str, str], cframe_path: str, mjd: int, drpver: Optional[str] = None,
                   from_sandbox: bool = True, direct: bool = True,
                   epochs_kind: str = 'calibration') -> Optional[Dict]:
    """Corrects the lvmFrames of one exposure for the large-scale flat-field error, in place.

    The correction is taken from the sky-line planes in the header of the exposure's
    lvmCFrame and the model files of its epoch, if they exist (see
    :func:`frame_coefficients`), and applied to every fiber of each channel's
    lvmFrame (see :func:`apply_frame_correction`), replacing any earlier correction.
    The steps after the lvmFrames (sky interpolation, resampling, flux calibration,
    joining, sky subtraction) have to be run again afterwards.

    Args:
        frame_paths: {channel: lvmFrame path} for b, r and z.
        cframe_path: the exposure's lvmCFrame, with the SKYPLANE keywords.
        mjd: MJD of the exposure, to find the model files.
        drpver: DRP version, for the calibration directories of a reduction.
        from_sandbox: look for the model files in the sandbox calibration directories.
        direct: use the exposure's own measurements (see :func:`frame_coefficients`).
        epochs_kind: where the model files are looked for (see :func:`find_flat_models`);
            by default the calibration epoch, since the model describes what the
            twilight flats of that epoch leave.

    Returns:
        The dict returned by :func:`frame_coefficients`, or None if nothing was corrected.
    """
    import os
    from astropy.io import fits
    from lvmdrp.core.rss import lvmFrame

    if not os.path.isfile(cframe_path):
        log.error(f'no lvmCFrame with the sky-line planes: {cframe_path}')
        return None
    header = fits.getheader(cframe_path)
    if 'SKYPLANE Z GX' not in header:
        log.error(f'no sky-line planes (SKYPLANE keywords) in {cframe_path}')
        return None
    missing = [p for p in frame_paths.values() if not os.path.isfile(p)]
    if len(frame_paths) < 3 or missing:
        log.error(f'lvmFrames missing: {missing or sorted(frame_paths)}')
        return None

    files = find_flat_models(mjd, drpver=drpver, from_sandbox=from_sandbox, epochs_kind=epochs_kind)
    models = {ch: read_flat_model(p) for ch, p in files.items()} if len(files) == 3 else None
    if models is None:
        log.info(f'no large-scale flat-field error model for MJD {mjd} (found {sorted(files)})')
    frames = {ch: lvmFrame.from_file(p) for ch, p in frame_paths.items()}
    previous = {ch: read_frame_correction(frames[ch]._header, ch) for ch in frames}
    nsci = int(np.sum(np.asarray(frames['b']._slitmap['telescope']).astype(str) == 'Sci'))
    result = frame_coefficients(header, models=models, previous=previous, direct=direct,
                                min_fibers=int(0.7 * MIN_FIBER_FRACTION * nsci))
    if result is None:
        return None
    info = {'RMODE': (result['rmode'], 'r: oh+bz, bz, 6364+bz or none'),
            'GUARD': (round(result['guard'], 3) if np.isfinite(result['guard']) else -1.0,
                      '|b - z| residual plane [%/10mm]'),
            'RGUARD': (round(result['rguard'], 3) if np.isfinite(result['rguard']) else -1.0,
                       '|r OH - mean(b, z)| plane [%/10mm]'),
            'EPOCH': (int(models['b']['epoch']) if models else -1, 'epoch of the model (-1: none)')}
    for ch, p in frame_paths.items():
        cinfo = dict(info)
        if models:
            cinfo[f'{ch.upper()} FILE'] = (os.path.basename(files[ch]), 'model file')
            if models[ch].get('modelid'):
                cinfo[f'{ch.upper()} MODELID'] = (models[ch]['modelid'], 'model identifier')
        apply_frame_correction(frames[ch], ch, result['coeffs'][ch], source=result['mode'], info=cinfo)
        frames[ch].writeFitsData(p)
    return result


# ----------------------------------------------------------------------------------------------
# Flats from the twilights: the instrument gradient of each epoch, with the twilight sky modelled
# ----------------------------------------------------------------------------------------------

# Sun altitude (degrees) at which the twilight sky term is zero: its amplitude grows linearly with depth below it
TWILIGHT_SUN_ALT_REF = -3.0


def twilight_planes(summary, channel: str) -> Dict[str, np.ndarray]:
    """The gradient measured in each twilight exposure, and its sky geometry.

    Args:
        summary: per-exposure twilight table (pandas DataFrame), as written by
            ``drp calibrations qa-twilights --trends``: columns expnum, channel,
            mjd, gcoeff1-4 (the gradient that stage 1 removed from the
            per-exposure flat), sun_alt and dir_sun (the direction towards the
            Sun across the science IFU, degrees from +x towards +y).
        channel: 'b', 'r' or 'z'.

    Returns:
        A dict of arrays over the usable exposures of the channel: expnum, mjd,
        gx and gy (% per 10 mm), depth (degrees of the Sun below
        `TWILIGHT_SUN_ALT_REF`), ux and uy (the unit vector towards the Sun).
    """
    tab = summary[summary['channel'].astype(str) == channel]
    # the coefficients are not normalized (they can be large and negative); only their ratios matter
    g = {k: np.asarray(tab[f'gcoeff{k}'], dtype=float) if f'gcoeff{k}' in tab else np.full(len(tab), np.nan)
         for k in (1, 2, 3, 4)}
    with np.errstate(divide='ignore', invalid='ignore'):
        gx, gy = 1000 * g[3] / g[1], 1000 * g[2] / g[1]
    sun_alt = _column(tab, 'sun_alt')
    dir_sun = np.radians(_column(tab, 'dir_sun'))
    mjd = _column(tab, 'mjd')
    ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(sun_alt) & np.isfinite(dir_sun) & np.isfinite(mjd)
    ok &= g[1] != 0
    return {'expnum': np.asarray(tab['expnum'], dtype=int)[ok], 'mjd': mjd[ok], 'gx': gx[ok], 'gy': gy[ok],
            'depth': TWILIGHT_SUN_ALT_REF - sun_alt[ok], 'ux': np.cos(dir_sun[ok]), 'uy': np.sin(dir_sun[ok])}


def fit_twilight_model(summary, epoch_mjds: Sequence[int], channels: str = 'brz',
                       clip: Optional[float] = None) -> Dict:
    """Fits the instrument gradient of each epoch to the twilight exposures.

    The gradient measured in each twilight exposure is modelled as a constant
    plane per epoch (the instrument gradient) plus a twilight sky gradient
    towards the Sun, with an amplitude that is linear in the depth of the Sun
    below `TWILIGHT_SUN_ALT_REF` (two coefficients per channel, shared by all
    epochs), as in the twilight model of the instrument-gradient QA. Of the variants tested
    on the 1.3.2 science sky lines, this one gave the flattest frames; a drift
    within an epoch, a focus term or clipping of outlying twilights did not
    help. With `clip`, exposures more than `clip` robust standard deviations
    from the fit are rejected and the fit repeated.

    Args:
        summary: per-exposure twilight table (see :func:`twilight_planes`).
        epoch_mjds: the first MJD of each epoch (the ffactor epochs).
        channels: the channels to fit.
        clip: clipping threshold, in robust standard deviations (default: no clipping).

    Returns:
        A dict with 'planes' ({channel: {epoch: (gx, gy)}}, % per 10 mm), 'sun'
        ({channel: (amplitude at the reference depth, amplitude per degree of
        depth)}, % per 10 mm), 'nexp'
        ({channel: {epoch: number of twilights}}), 'rejected' ({channel: expnums})
        and 'rms' ({channel: rms residual, % per 10 mm}).

    Raises:
        ValueError: if a channel has no usable twilight exposures.
    """
    out = {'planes': {}, 'sun': {}, 'nexp': {}, 'rejected': {}, 'rms': {}}
    for ch in channels:
        tw = twilight_planes(summary, ch)
        n = tw['mjd'].size
        if n == 0:
            raise ValueError(f'channel {ch}: no usable twilight exposures')
        epoch = assign_epochs(tw['mjd'], epoch_mjds)
        epochs = sorted(set(int(e) for e in epoch))
        use = np.ones(n, dtype=bool)
        for _ in range(2 if clip else 1):
            cols = []
            for e in epochs:
                sel = (epoch == e).astype(float)
                cols.append(np.r_[sel, np.zeros(n)])
                cols.append(np.r_[np.zeros(n), sel])
            cols.append(np.r_[tw['ux'], tw['uy']])
            cols.append(np.r_[tw['depth'] * tw['ux'], tw['depth'] * tw['uy']])
            A = np.column_stack(cols)
            T = np.r_[tw['gx'], tw['gy']]
            rows = np.r_[use, use]
            coef, *_ = np.linalg.lstsq(A[rows], T[rows], rcond=None)
            res = T - A @ coef
            dist = np.hypot(res[:n], res[n:])
            sig = 1.4826 * np.median(np.abs(dist[use] - np.median(dist[use])))
            new = dist < np.median(dist[use]) + clip * sig if (clip and sig > 0) else use
            if np.array_equal(new, use):
                break
            use = new
        out['planes'][ch] = {e: (float(coef[2 * i]), float(coef[2 * i + 1])) for i, e in enumerate(epochs)}
        out['sun'][ch] = (float(coef[-2]), float(coef[-1]))
        out['nexp'][ch] = {e: int(np.sum(use & (epoch == e))) for e in epochs}
        out['rejected'][ch] = sorted(int(x) for x in tw['expnum'][~use])
        out['rms'][ch] = float(np.sqrt(np.mean(dist[use] ** 2)))
    return out


def _write_flat(mflat, out_path: str, in_path: Optional[str] = None) -> None:
    """Writes a flat, replacing a symbolic link at `out_path` (never writing through it).

    The FRAMES extension of `in_path` (the flat that was read; default `out_path`) is
    carried over, since the RSS writer does not keep it.
    """
    import os
    from lvmdrp.core.header import read_frames_extension, write_frames_extension
    src = in_path or out_path
    frames = read_frames_extension(src) if os.path.isfile(src) else None
    if os.path.islink(out_path):
        os.remove(out_path)
    mflat.writeFitsData(out_path)
    if frames is not None:
        write_frames_extension(out_path, frames)


def apply_twilight_gradient(in_mflat: str, gx: float, gy: float, epoch: int,
                            out_mflat: Optional[str] = None, write_output: bool = False):
    """Puts the instrument gradient of the twilights into a master twilight fiber flat.

    The science fibers of the flat are multiplied by the plane
    ``1 + (gx * x + gy * y) / 1000`` (normalized to a mean of 1 over them); the
    other fibers are not changed. The plane is recorded in the header
    (``<CH> FIBERFLAT TWIGRAD``, ``TWIGX``, ``TWIGY``, ``TWIEPOCH``); a flat that
    already has it is left as it is.

    Args:
        in_mflat: the master twilight fiber flat (lvm-mfiberflat_twilight-<channel>.fits).
        gx: x slope of the plane, % per 10 mm.
        gy: y slope of the plane, % per 10 mm.
        epoch: the epoch the plane was fitted for, recorded in the header.
        out_mflat: where to write the result (default: `in_mflat`); a symbolic
            link there is replaced by the file, never written through.
        write_output: write the result.

    Returns:
        The flat (an RSS) and whether it was changed.
    """
    from lvmdrp.core.rss import RSS
    mflat = RSS.from_file(in_mflat)
    ch = str(mflat._header['CCD'])[0].upper()
    out_mflat = out_mflat or in_mflat
    if mflat._header.get(f'{ch} FIBERFLAT TWIGRAD', False):
        log.info(f'{in_mflat}: the twilight gradient is already in the flat, nothing to do')
        return mflat, False
    sci = np.flatnonzero(np.asarray(mflat._slitmap['telescope']).astype(str) == 'Sci')
    x = np.asarray(mflat._slitmap['xpmm'], dtype=float)[sci]
    y = np.asarray(mflat._slitmap['ypmm'], dtype=float)[sci]
    plane = 1 + (gx * x + gy * y) / 1000
    plane /= np.mean(plane)
    mflat._data[sci] = mflat._data[sci] * plane[:, None]
    if mflat._error is not None:
        mflat._error[sci] = mflat._error[sci] * plane[:, None]
    mflat.setHdrValue(f'HIERARCH {ch} FIBERFLAT TWIGRAD', True, 'twilight instrument gradient in flat (#268)')
    mflat.setHdrValue(f'HIERARCH {ch} FIBERFLAT TWIGX', round(float(gx), 4), 'twilight gradient x [%/10mm]')
    mflat.setHdrValue(f'HIERARCH {ch} FIBERFLAT TWIGY', round(float(gy), 4), 'twilight gradient y [%/10mm]')
    mflat.setHdrValue(f'HIERARCH {ch} FIBERFLAT TWIEPOCH', int(epoch), 'ffactor epoch of the gradient')
    if write_output:
        log.info(f'writing {out_mflat} with the twilight gradient ({gx:.2f}, {gy:.2f}) %/10mm')
        _write_flat(mflat, out_mflat, in_path=in_mflat)
    return mflat, True


def undo_twilight_gradient(in_mflat: str, write_output: bool = False):
    """Takes the twilight gradient recorded by :func:`apply_twilight_gradient` out of a master flat.

    Args:
        in_mflat: the master twilight fiber flat.
        write_output: overwrite `in_mflat` with the result.

    Returns:
        The flat (an RSS) and whether it was changed.
    """
    from lvmdrp.core.rss import RSS
    mflat = RSS.from_file(in_mflat)
    ch = str(mflat._header['CCD'])[0].upper()
    if not mflat._header.get(f'{ch} FIBERFLAT TWIGRAD', False):
        log.info(f'{in_mflat}: no twilight gradient in the flat, nothing to undo')
        return mflat, False
    gx, gy = float(mflat._header[f'{ch} FIBERFLAT TWIGX']), float(mflat._header[f'{ch} FIBERFLAT TWIGY'])
    sci = np.flatnonzero(np.asarray(mflat._slitmap['telescope']).astype(str) == 'Sci')
    x = np.asarray(mflat._slitmap['xpmm'], dtype=float)[sci]
    y = np.asarray(mflat._slitmap['ypmm'], dtype=float)[sci]
    plane = 1 + (gx * x + gy * y) / 1000
    plane /= np.mean(plane)
    mflat._data[sci] = mflat._data[sci] / plane[:, None]
    if mflat._error is not None:
        mflat._error[sci] = mflat._error[sci] / plane[:, None]
    mflat.setHdrValue(f'HIERARCH {ch} FIBERFLAT TWIGRAD', False, 'twilight instrument gradient in flat (#268)')
    if write_output:
        log.info(f'writing {in_mflat} without the twilight gradient')
        _write_flat(mflat, in_mflat)
    return mflat, True


def link_calibration_tree(src_root: str, dst_root: str) -> int:
    """Makes a copy of a calibration directory tree made of symbolic links.

    Every file under `src_root` gets a symbolic link at the same place under
    `dst_root` (directories are created); existing entries are left as they are.
    A calibration file can then be replaced in the new tree, by removing its link
    and writing the file, without touching the original.

    Args:
        src_root: the calibration directory (e.g. $LVM_MASTER_DIR).
        dst_root: the new calibration directory.

    Returns:
        The number of links made.
    """
    import os
    src_root, dst_root = os.path.abspath(src_root), os.path.abspath(dst_root)
    if dst_root == src_root or dst_root.startswith(src_root + os.sep):
        raise ValueError(f'the new calibration directory {dst_root} must be outside {src_root}')
    nlinks = 0
    for dirpath, _, filenames in os.walk(src_root):
        target = os.path.join(dst_root, os.path.relpath(dirpath, src_root))
        os.makedirs(target, exist_ok=True)
        for name in filenames:
            dst = os.path.join(target, name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(dirpath, name), dst)
                nlinks += 1
    return nlinks


def create_twilight_gradients(summary_path: str, ffactor_epochs: Dict, calibration_epochs: Dict,
                              channels: str = 'brz', out_calib_dir: Optional[str] = None, in_place: bool = False,
                              drpver: Optional[str] = None, from_sandbox: bool = True,
                              undo: bool = False, dry_run: bool = False) -> Dict:
    """Fits the twilight instrument gradient of each ffactor epoch and puts it into the master twilight flats.

    Every calibration epoch gets the plane of the ffactor epoch it belongs to.
    By default the original flats are not changed: with `out_calib_dir`, a new
    calibration tree is made there (symbolic links to every file of the sandbox
    calibration directory, see :func:`link_calibration_tree`) and the new flats
    replace their links, so the old and the new calibrations can both be used
    (``LVM_MASTER_DIR``). Changing the original flats needs `in_place`.

    Args:
        summary_path: per-exposure twilight table (CSV), from ``drp calibrations qa-twilights --trends``.
        ffactor_epochs: the ffactor epochs.
        calibration_epochs: the calibration epochs, whose master flats get the gradient.
        channels: the channels to process.
        out_calib_dir: the new calibration directory (sandbox layout).
        in_place: change the original flats instead.
        drpver: DRP version, for the calibration directories of a reduction (with `in_place`).
        from_sandbox: use the sandbox calibration directory ($LVM_MASTER_DIR) rather than those of `drpver`.
        undo: take the twilight gradient out of the flats instead (in `out_calib_dir`, or in place).
        dry_run: fit and log, but change no file.

    Returns:
        The fit (see :func:`fit_twilight_model`), or {} with `undo`.

    Raises:
        ValueError: without `out_calib_dir` or `in_place`, or with both, or with
            `out_calib_dir` and the calibrations of a reduction.
    """
    import os
    import pandas as pd
    from lvmdrp.core.constants import get_masters_dir
    from lvmdrp.utils.paths import get_calib_paths

    if (out_calib_dir is None) == (not in_place):
        raise ValueError('give either a new calibration directory (out_calib_dir) or in_place, not both')
    if out_calib_dir is not None and not from_sandbox:
        raise ValueError('a new calibration directory can only be made from the sandbox calibrations')
    masters_dir = get_masters_dir()

    fit = {}
    if not undo:
        summary = pd.read_csv(summary_path)
        fit = fit_twilight_model(summary, sorted(ffactor_epochs), channels=channels)
        for ch in channels:
            sun0, sun1 = fit['sun'][ch]
            log.info(f'channel {ch}: twilight sky term {sun0:.2f} + {sun1:.3f} x depth %/10mm, '
                     f'rms {fit["rms"][ch]:.2f}')
            for e, (gx, gy) in fit['planes'][ch].items():
                log.info(f'  epoch {e}: gradient ({gx:.2f}, {gy:.2f}) %/10mm from {fit["nexp"][ch][e]} twilights')
    if out_calib_dir is not None and not dry_run:
        n = link_calibration_tree(masters_dir, out_calib_dir)
        log.info(f'new calibration tree {out_calib_dir}: {n} links to {masters_dir}')

    for cal in sorted(calibration_epochs):
        epoch = int(assign_epochs([cal], sorted(ffactor_epochs))[0])
        flats = get_calib_paths(mjd=cal, version=drpver, flavors={'fiberflat_twilight'},
                                from_sandbox=from_sandbox).get('fiberflat_twilight', {})
        for ch in channels:
            src = flats.get(ch)
            if src is None or not os.path.exists(src):
                log.warning(f'master twilight flat for calibration epoch {cal}, channel {ch} not found, skipping')
                continue
            dst = src if in_place else os.path.join(out_calib_dir, os.path.relpath(src, masters_dir))
            if undo:
                log.info(f'{"would take" if dry_run else "taking"} the twilight gradient out of {dst}')
                if not dry_run:
                    undo_twilight_gradient(dst, write_output=True)
                continue
            if epoch not in fit['planes'][ch]:
                log.warning(f'no twilights for ffactor epoch {epoch} (calibration epoch {cal}), channel {ch}, skipping')
                continue
            gx, gy = fit['planes'][ch][epoch]
            log.info(f'{"would put" if dry_run else "putting"} ({gx:.2f}, {gy:.2f}) %/10mm into {dst}')
            if not dry_run:
                apply_twilight_gradient(src, gx, gy, epoch, out_mflat=dst, write_output=True)
    return fit
