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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

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
              percentiles: Tuple[float, float] = STAR_PERCENTILES) -> np.ndarray:
    """Flags fibers dominated by stars or other bright sources.

    Fibers whose median continuum near `STAR_CONTINUUM[arm]` is outside the
    given percentiles of all fibers, or not finite, are flagged.

    Args:
        flux: fluxes, fibers x wavelength.
        wave: wavelength of each pixel, Angstrom.
        arm: 'b', 'r' or 'z'.
        percentiles: lower and upper percentile of the continuum kept.

    Returns:
        True for the fibers to reject.
    """
    cwave = STAR_CONTINUUM[arm]
    sel = _window(wave, cwave - STAR_CONTINUUM_HALF_WIDTH, cwave + STAR_CONTINUUM_HALF_WIDTH)
    with np.errstate(all='ignore'):
        cont = np.nanmedian(flux[:, sel].astype(float), axis=1)
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


def measure_cframe(rss, write_header: bool = True, **kwargs) -> Dict[str, Optional[Dict[str, float]]]:
    """Measures the sky-line planes of a channel-combined frame (lvmCFrame).

    Args:
        rss: an :class:`lvmdrp.core.rss.lvmCFrame` (or any RSS with a 1D
            wavelength array and a slitmap).
        write_header: write the results to the header of `rss`
            (see :func:`skyline_plane_cards`).
        **kwargs: passed to :func:`measure_skyline_planes`.

    Returns:
        The dict returned by :func:`measure_skyline_planes`.
    """
    if rss._wave is None or np.ndim(rss._wave) != 1:
        raise ValueError('measure_cframe needs a frame with a 1D wavelength array')
    results = measure_skyline_planes(rss._data, rss._wave, rss._slitmap, **kwargs)
    if write_header:
        for kw, value, comment in skyline_plane_cards(results):
            rss.setHdrValue(kw, value, comment)
    return results
