"""Rescale the predicted sky emission lines on the spectrum being sky-subtracted.

Why this works
--------------
The prediction transfers the sky lines from the sky arms to the science
pointing.  The line SHAPES transfer well, but their BRIGHTNESS does not: the OH
emission differs by a few percent between the science and sky directions, and
across the IFU.  OH brightness is most of the flux-error tail.  A sky line,
though, is 2-3 A wide and stands out from whatever science signal lies under
it.  So its brightness can be measured on the very spectrum it will be
subtracted from, while the broad science continuum, which the sky model cannot
separate from the target, is left alone.

The model
---------
The sky lines are split into a few templates built from the PREDICTED
coefficients (:func:`line_templates`):

* one per OH vibrational band v' (each OH coefficient is the population of one
  upper level v', N', F', so a band is a group of coefficients);
* optionally one rotational-temperature tilt per band: the same lines weighted
  by (E_upper - <E_upper>), so a temperature difference between science and
  sky is a linear term;
* one per atomic line family (ATOM_Og [O I] 5577, ATOM_Or [O I] 6300/6364,
  ATOM_Na Na D, ...), and the O2 b band.

On the science spectrum, the residual ``r = observed - sky`` is modelled as

    HP(r) = sum_b  d_b * HP(T_b)  +  noise  +  science features

where ``HP`` is the same LINEAR high-pass (x minus its running mean over
``highpass_A``) applied to data and templates alike.  The science continuum is
broad, so HP removes it; any line leakage into the running mean happens
identically to the templates, so the model stays exact.  ``d_b`` is the
fractional brightness change of template b (``s_b = 1 + d_b``).

The fit is weighted least squares with two safeguards:

* robust (Huber) iteration on standardised residuals, so that narrow science
  features (stellar absorption lines, unmasked emission lines) are
  down-weighted instead of pulling the scales;
* a Gaussian prior on every ``d_b`` (``prior_sigma``), so that a template with
  little signal on this row stays at the prediction.

The correction ``sum_b d_b * T_b`` (unfiltered templates) is ADDED to the
predicted sky, just like :func:`~mlp_predictor.sky_arm_correction.sky_arm_residual_correction`.

Protecting the science signal
-----------------------------
Tested by injecting PHOENIX G2 V, K3 V, K3 III, M2 V and M3 III spectra at
0.3-10x the sky continuum, and 15 red nebular lines at 0.5x and 3x the sky
continuum.  Velocities were random within +-150 km/s (stars) and +-300 km/s
(lines), on 120 uncrowded held-out rows.  Four measures are needed, and all are
defaults:

* ``FIXED_FAMILIES`` keep their predicted brightness.  Science [O I]
  6300/6364 coincides with the sky line: rescaled, it lost up to 100% of its
  flux.  Na D and K I are stellar absorption lines (5x G star: Na D window
  chi2 0.16 -> 5.3).  N I 5199 sits on stellar Mg b: its scale moved by up to
  0.5 under a bright star.
* The callers mask ``NEBULAR_LINES_RED`` as well as the decomposition's
  science lines, slid by the measured Halpha velocity and widened by
  ``UNMEASURED_VELOCITY_KM_S`` when there is none.  Unmasked, weak lines on
  OH lines lost a median of 1-2% and up to 8%.
* ``science_noise`` adds 10% of the local science continuum to the noise.
  Stellar lines survive any high-pass, because they are as narrow as the sky
  lines; this lowers their pull.
* No chi2 guard (``guard=False``): see its parameter note.

With these, the recovered flux of every nebular line changes by <= 0.6% at
the median (5th percentile >= -4%) at 0.5x, and by <= 0.3% at 3x.  A star
changes the correction by 0.14-0.6% of its own flux (RMS in the red).  The OH
band scales move by <= 0.03 at 1x and <= 0.16 at 10x the sky continuum.

Measured (600 held-out rows of the palacecorr corpus, after the full-band
sky-arm correction, median single-fibre photon chi2 on the decomposition's
science-line mask):

    red >= 6000 A     0.460 -> 0.301      decomposition's own fit 0.725
    OH-line pixels    1.187 -> 0.203                              0.614
    full band         0.364 -> 0.270                              0.931

* The fitted OH scales match the science decomposition's own band
  brightness: the median error is 0.1-0.6%, against the 1.5-2.7% the
  prediction misses.
* The result barely depends on the settings: high-pass 10-50 A, prior
  0.03-0.3 and Huber on/off agree to 0.003.  The tilt helps on OH pixels
  (0.223 -> 0.203); a single scale for all OH is clearly worse (0.293).
* The science protections cost red 0.284 -> 0.301, mostly because sky
  [O I] 6300/6364 is no longer rescaled.  They leave OH pixels unchanged.
* On 8.0% of rows the raw red chi2 rises, by at most 7.6%.  This is NOT line
  damage.  Those rows already carry a broad positive offset between the lines
  (+2.1% of the sky continuum: continuum light, mostly on crowded fields).
  It meets the small broad part of the correction (line wings, blends), whose
  sign is opposite.  With broad offsets removed (100 A high-pass), the same
  48 rows improve from 0.275 to 0.143, and only 1.5% of all rows get worse.
  Confining the correction to the line regions lowers the raw count but
  throws away real line correction: line-scale chi2 0.147 -> 0.197 at
  lines > 1x continuum.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

__all__ = ["oh_group_labels", "line_templates", "sky_line_scaling_correction",
           "NEBULAR_LINES_RED", "UNMEASURED_VELOCITY_KM_S", "FIXED_FAMILIES"]

# Nebular lines that fall among the sky lines and are NOT in the decomposition's
# science-line mask.  Unmasked, a weak one sitting on an OH line is partly
# absorbed into that band's scale.  Callers mask them, slid by the measured
# Halpha velocity: `data.science_line_mask_rows(..., extra_lines=NEBULAR_LINES_RED,
# widen_if_unmeasured_km_s=UNMEASURED_VELOCITY_KM_S)`.
NEBULAR_LINES_RED = (
    ("HeI5876", 5875.62), ("[OI]6300", 6300.30), ("[OI]6364", 6363.78),
    ("HeI6678", 6678.15), ("HeI7065", 7065.19), ("[ArIII]7136", 7135.79),
    ("[FeII]7155", 7155.16), ("HeI7281", 7281.35), ("[OII]7320", 7319.99),
    ("[OII]7330", 7330.20), ("[NiII]7378", 7377.83), ("[ArIII]7751", 7751.06),
    ("P16", 8502.48), ("P15", 8545.38), ("P14", 8598.39), ("[FeII]8617", 8616.95),
    ("P13", 8665.02), ("P12", 8750.47), ("P11", 8862.78), ("P10", 9014.91),
    ("[SIII]9069", 9068.60), ("P9", 9229.01), ("[SIII]9531", 9530.60),
    ("P8", 9545.97),
)
# Widening of those windows when Halpha is too weak for a velocity.
UNMEASURED_VELOCITY_KM_S = 150.0
# Families that keep their predicted brightness: each coincides with a science
# feature that the fit cannot tell apart from the sky line.  Na D and K I are
# stellar absorption lines, [O I] 6300/6364 is nebular emission at the sky
# wavelength, and N I 5199 (0.1% of the line flux) sits on stellar Mg b / MgH.
FIXED_FAMILIES = ("ATOM_Na", "ATOM_K", "ATOM_Or", "ATOM_N")

_E_SCALE = 1000.0          # cm^-1: the tilt regressor's unit of upper-level energy


def oh_group_labels(model):
    """``(v_upper, E_upper)`` per OH coefficient, in the basis order of ``model``.

    Rebuilds the grouping ``sky_decomp.fit._oh_line_catalog`` uses -- same
    table, same wavelength cut, same keys -- and reads the vibrational level and
    the upper-level energy (cm^-1) of each group.  Checked against the model's
    own line groups, so a change in the catalog cannot silently mislabel bands.
    """
    from sky_decomp.fit import CAP_WAVE, decode_hitran_id, read_static_table, vac_to_air
    oh = read_static_table(str(model._pmd_path("pmd_popmodel_OH.dat")))
    oh["wave"] = vac_to_air(np.asarray(oh["lam"], float) * 1e4)
    lo, hi = float(model.wave.min()), float(model.wave.max())
    oh = decode_hitran_id(oh[(oh["wave"] >= lo - CAP_WAVE) & (oh["wave"] <= hi + CAP_WAVE)])
    groups = oh.group_by(list(model.oh_group_keys)).groups
    v = np.asarray([int(g["v_upper"][0]) for g in groups])
    e = np.asarray([float(np.mean(g["Ei"])) for g in groups])
    ref = getattr(model, "_oh_line_groups", None)
    if ref is not None:
        if len(ref) != len(groups):
            raise ValueError(f"OH catalog has {len(groups)} groups, the model {len(ref)}")
        for (wl, _), g in zip(ref, groups):
            if not np.allclose(np.sort(wl), np.sort(np.asarray(g["wave"], float))):
                raise ValueError("OH catalog grouping does not match the model's line groups")
    return v, e


def line_templates(model, mats, coef, *, tilt=True, min_band_fraction=0.01,
                   atoms=True, o2=True, exclude=FIXED_FAMILIES):
    """Per-family sky-line templates from ONE row's predicted coefficients.

    ``model`` is the row's decomposer and ``mats`` its assembled matrices (as
    used for the reconstruction), so the templates carry the same LSF and
    telluric transmission as the predicted sky.  Returns ``{name: (n_wave,)}``
    in the units of the reconstruction.  Bands holding less than
    ``min_band_fraction`` of the predicted OH flux are merged into their
    neighbour, so that every template has lines to fit.  Families named in
    ``exclude`` get no template, so they keep their predicted brightness (see
    the module notes on Na D).
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    sl = model._component_slices(mats)
    m_oh = np.asarray(mats["oh"], dtype=np.float64)
    c_oh = coef[sl["oh"]]
    v, e = oh_group_labels(model)
    if v.size != m_oh.shape[0]:
        raise ValueError(f"{v.size} OH labels for {m_oh.shape[0]} OH basis rows")
    flux_g = np.clip(c_oh, 0.0, None) * m_oh.sum(axis=1)       # flux carried by each group
    total = float(flux_g.sum())
    bands = sorted(set(v.tolist()))
    # Merge faint bands into the nearest brighter one.
    keep = [b for b in bands if total > 0 and flux_g[v == b].sum() >= min_band_fraction * total]
    if not keep:
        keep = bands[:1]
    assign = {b: min(keep, key=lambda k: (abs(k - b), k)) for b in bands}
    out = {}
    for k in keep:
        idx = np.flatnonzero(np.isin(v, [b for b in bands if assign[b] == k]))
        out[f"OH_v{k}"] = m_oh[idx].T @ c_oh[idx]
        if tilt:
            w = flux_g[idx]
            e_mean = float(np.sum(w * e[idx]) / w.sum()) if w.sum() > 0 else float(np.mean(e[idx]))
            out[f"OH_v{k}_tilt"] = m_oh[idx].T @ (c_oh[idx] * (e[idx] - e_mean) / _E_SCALE)
    if atoms:
        m_at = np.asarray(mats["atom"], dtype=np.float64)
        c_at = coef[sl["atom"]]
        for j, name in enumerate(model.atom_names):
            if c_at[j] > 0 and name not in exclude:
                out[name] = m_at[j] * c_at[j]
    if o2 and "o2" in sl:
        c_o2 = coef[sl["o2"]]
        if np.any(c_o2 > 0):
            out["O2"] = np.asarray(mats["o2"], dtype=np.float64).T @ c_o2
    return out


def _highpass(x, mask, width_px):
    """x minus its masked running mean: linear in x for a fixed mask."""
    m = mask.astype(np.float64)
    num = uniform_filter1d(np.where(mask, x, 0.0), width_px, mode="nearest")
    den = uniform_filter1d(m, width_px, mode="nearest")
    return np.where(mask, x - num / np.where(den > 0, den, 1.0), 0.0)


def sky_line_scaling_correction(
    wave,
    sci_observed,
    sky_model,
    templates: Mapping[str, np.ndarray],
    *,
    variance=None,
    mask=None,
    highpass_A: float = 25.0,
    prior_sigma: float = 0.1,
    science_noise: float = 0.1,
    science_noise_width_A: float = 100.0,
    huber_k: float = 2.0,
    robust: str = "huber",
    tukey_c: float = 4.685,
    n_iter: int = 8,
    line_fraction: float = 0.02,
    guard: bool = False,
    return_info: bool = False,
):
    """Correction to ADD to the predicted sky, from rescaling its sky lines.

    Parameters
    ----------
    wave : (n_wave,) wavelength grid in Angstrom.
    sci_observed : the science spectrum being sky-subtracted (one fibre, or here
        the science median), (n_wave,).
    sky_model : the sky that will be subtracted from it before this step: the
        prediction, plus the sky-arm correction when that is applied.
    templates : ``{name: (n_wave,)}`` from :func:`line_templates`, in the units
        of ``sky_model``.
    variance : photon variance of ``sci_observed`` (only its SHAPE matters: the
        noise scale is measured from the residual).  None = uniform.
    mask : True where pixels must not be used (the science-line mask).
    highpass_A : width of the running mean the high-pass removes.  Wide enough
        to keep a line and its wings (~10x the LSF FWHM), narrow enough that the
        science continuum under the lines is gone.
    prior_sigma : 1-sigma prior on each fractional brightness change.  The tilt
        terms use the same width per 1000 cm^-1 of upper-level energy.
    science_noise : treat this fraction of the local science continuum as
        extra noise, added in quadrature to ``variance``.  The narrow
        structure of a science continuum (stellar absorption lines, band heads)
        survives any high-pass, because it is as narrow as the sky lines; its
        amplitude scales with the continuum, so a bright star lowers the
        weight of the data against the prior instead of leaking into the
        scales.  The continuum is the running median of ``sci_observed -
        sky_model`` over ``science_noise_width_A``, floored at zero.  It needs
        an ABSOLUTE ``variance`` and is skipped without one.
    huber_k : Huber threshold, in robust standard deviations.
    robust : ``'huber'`` (down-weights large residuals) or ``'tukey'`` (biweight
        with threshold ``tukey_c``: rejects them entirely).
    line_fraction : pixels where the high-passed templates reach this fraction
        of their row's 99th percentile define the robust noise scale.
    guard : return no correction when it raises the variance-weighted chi2 of
        ``sci_observed - sky_model`` over the good pixels.  OFF by default
        because it is NOT safe with science light: that chi2 includes the
        science continuum, which couples to the broad part of the correction,
        so a star of 0.3-10x the sky continuum flipped its decision on 40-75%
        of rows.  Use it only on spectra known to be sky-dominated.  The raw
        chi2 rises it was added for are broad-offset effects, not line
        damage (see the module notes).
    return_info : also return ``{name: scale}``, its 1-sigma, and fit stats.
    """
    wave = np.asarray(wave, dtype=np.float64)
    y_obs = np.asarray(sci_observed, dtype=np.float64)
    sky = np.asarray(sky_model, dtype=np.float64)
    names = [k for k, t in templates.items() if np.any(np.asarray(t) != 0)]
    if not names:
        zero = np.zeros_like(sky)
        return (zero, dict(scales={}, sigma={}, n_pix=0, accepted=False)) if return_info else zero
    T = np.vstack([np.asarray(templates[k], dtype=np.float64) for k in names])
    if T.shape[1] != wave.size or y_obs.shape != wave.shape or sky.shape != wave.shape:
        raise ValueError("wave, sci_observed, sky_model and templates must share one grid")
    good = np.isfinite(y_obs) & np.isfinite(sky) & np.all(np.isfinite(T), axis=0)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    w0 = np.ones_like(wave)
    if variance is not None:
        var = np.asarray(variance, dtype=np.float64)
        if science_noise:
            from scipy.ndimage import median_filter
            _r = np.where(np.isfinite(y_obs - sky), y_obs - sky, 0.0)
            _px = max(3, int(round(science_noise_width_A / float(np.median(np.diff(wave))))) | 1)
            cont_sci = np.clip(median_filter(_r, size=_px, mode="nearest"), 0.0, None)
            var = var + (float(science_noise) * cont_sci) ** 2
        good &= np.isfinite(var) & (var > 0)
        w0 = np.where(good, 1.0 / np.where(good, var, 1.0), 0.0)
        w0 /= np.median(w0[good]) if good.any() else 1.0
    px = max(3, int(round(highpass_A / float(np.median(np.diff(wave))))) | 1)
    y = _highpass(y_obs - sky, good, px)
    X = np.vstack([_highpass(t, good, px) for t in T])
    strength = np.sum(np.abs(X), axis=0)
    line_pix = good & (strength > line_fraction * np.percentile(strength[good], 99)) if good.any() else good
    n_line = int(line_pix.sum())
    if n_line < 3 * len(names):
        zero = np.zeros_like(sky)
        return (zero, dict(scales={k: 1.0 for k in names}, sigma={k: np.inf for k in names},
                           n_pix=n_line, accepted=False)) if return_info else zero

    # Prior precision in data units: the residual's robust scale sets the noise
    # unit, so the prior's pull is the same whatever the flux units.
    hub = np.ones_like(wave)
    d = np.zeros(len(names))
    for _ in range(int(n_iter)):
        e = y - d @ X
        z = np.sqrt(w0) * e
        s = 1.4826 * np.median(np.abs(z[line_pix])) or 1.0
        u = np.abs(z) / s
        if robust == "tukey":
            hub = np.where(u < tukey_c, (1.0 - (u / tukey_c) ** 2) ** 2, 0.0)
        else:
            hub = huber_k / np.maximum(u, huber_k)      # 1 inside the threshold
        w = np.where(good, w0 * hub, 0.0) / s ** 2
        A = (X * w) @ X.T + np.eye(len(names)) / prior_sigma ** 2
        b = (X * w) @ y
        d_new = np.linalg.solve(A, b)
        if np.max(np.abs(d_new - d)) < 1e-5:
            d = d_new
            break
        d = d_new
    corr = d @ T
    accepted = True
    if guard:
        r0 = np.where(good, y_obs - sky, 0.0)
        c2_before = float(np.sum(w0 * r0 ** 2))
        c2_after = float(np.sum(w0 * np.where(good, r0 - corr, 0.0) ** 2))
        accepted = c2_after <= c2_before
        if not accepted:
            corr = np.zeros_like(corr)
    if not return_info:
        return corr
    cov = np.linalg.inv(A)
    info = dict(scales={k: 1.0 + float(v) for k, v in zip(names, d)},
                sigma={k: float(np.sqrt(cov[i, i])) for i, k in enumerate(names)},
                n_pix=n_line, noise_scale=float(s),
                downweighted=float(np.mean(hub[line_pix] < 1.0)),
                accepted=accepted)
    return corr, info
