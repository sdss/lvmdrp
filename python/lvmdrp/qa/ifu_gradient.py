# encoding: utf-8
"""Instrument gradient across the science IFU.

The gradient that the pipeline fits across the science IFU (the plane of
:class:`lvmdrp.core.fit_profile.IFUGradient`) mixes two components:

- an instrument illumination gradient I, fixed in some instrument frame;
- a sky gradient S, fixed on the sky: towards the Sun in twilights, towards the
  horizon (airglow) and the Moon at night.

The twilight fiber flats remove the whole plane, so I never reaches the master
flats and stays in the flat-fielded science frames. This module tests whether I
exists and characterizes it before any pipeline change: which frame holds it
fixed, whether it jumps at instrument interventions, and whether it agrees
between twilights, dome flats and science frames.

For one channel, the gradient vector g_e of exposure e (at the IFU edge, in IFU
coordinates, %) is modeled as

    g_e = R(s phi_e) I_k + sum_j a_j(e) u_j(e) + noise

with I_k the instrument gradient of the segment k (between instrument
interventions) the exposure belongs to, R(s phi_e) a rotation by the K-mirror
angle for an instrument gradient fixed before the K-mirror (identity for one
fixed in the IFU), u_j(e) the unit vectors of the sky directions (Sun, horizon,
Moon) from the geometry of the exposure, and a_j(e) their amplitudes: constant,
linear in the depression of the Sun, or free per night or per exposure. The
model is linear and is fitted by least squares; competing models are compared
with the BIC and with the rms of leave-one-night-out predictions.

The inputs are the tables of the trends dashboards
(:func:`lvmdrp.qa.fiberflats.qa_twilight_trends` and
:func:`~lvmdrp.qa.fiberflats.qa_dome_trends`) and, for science frames, the
summary table of the flat field QA (:func:`lvmdrp.qa.fiberflats.run_flatfield_qa_batch`)
with the geometry of each frame (:func:`science_gradients`). The dashboard is
written by :func:`qa_instrument_gradient` (``drp calibrations qa-instrument-gradient``).
"""

import json
import os
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from html import escape
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from lvmdrp import log
from lvmdrp.qa.fiberflats import (
    _header_float, _night_sjd, _rayleigh_pvalue, _resultant_length, _wrap_angle, frame_gradients, load_calibration_epochs,
    load_kmirror_relation, siderostat_field_angle, twilight_geometry,
)
from lvmdrp.qa.report import DIVERGING, DIVERGING_DARK, SEQUENTIAL, SERIES, THEME, base_layout, html_table, sandbox_qa_dir, style_axes, write_dashboard
from lvmdrp.utils.convert import tileid_grp


# frames that can hold the instrument gradient fixed: (label, rotation sign applied to the K-mirror angle)
INSTRUMENT_FRAMES = {
    "none": ("no instrument gradient", None),
    "ifu": ("fixed in the IFU", 0),
    "kmirror": ("fixed before the K-mirror, rotating with +phi", 1),
    "kmirror-": ("fixed before the K-mirror, rotating with -phi", -1),
}

# amplitude of a sky gradient along its direction
SKY_AMPLITUDES = ("constant", "depth", "night", "exposure")

# default sky gradients of each source: {direction column: amplitude}
DEFAULT_SKY = {
    "twilight": {"dir_sun": "depth"},
    "dome": {"dir_horizon": "constant"},
    "science": {"dir_horizon": "exposure"},
}

# triggers of the calibration epochs that may change the instrument
INTERVENTION_TRIGGERS = ("instrument intervention", "warm-up", "warm up")

# layout of the sections of the dashboard: each source and channel is a <details> whose content is one grid
# that can shrink to the page width, with tables and charts side by side on wide screens
PAGE_STYLE = """<style>
details.ig { display: block; min-width: 0; }
details.ig summary { font-size: 15px; color: var(--ink); }
.ig-body { display: grid; gap: 12px; min-width: 0; }
.ig-body h3 { margin: 12px 0 0; font-size: 16px; font-weight: 600; }
.ig-body td { white-space: normal; }
.ig-body td.num { white-space: nowrap; }
.ig-body th, .ig-body td { padding: 6px 9px; }
.ig-pair { display: grid; gap: 12px; align-items: start; min-width: 0; }
@media (min-width: 980px) { .ig-pair { grid-template-columns: minmax(0, 11fr) minmax(0, 9fr); } }
.ig-pair > * { min-width: 0; }
</style>"""

# definitions of the quantities of the dashboard, shown at the bottom of the page
DEFINITIONS = r"""<section id="definitions">
  <h2>How the quantities are computed</h2>
  <p>Exposures are indexed by \(e\), nights by \(\nu\) and segments between breaks by \(k(e)\). Gradients are vectors in IFU
  coordinates, in %, and directions are measured from the IFU \(+x\) axis towards \(+y\). The fits are done separately for each
  channel and for each source<span data-sources="joint"> and for all of them together (joint)</span>.</p>
  <div class="defs">
    <div class="def">
      <h3>Measured gradient</h3>
      <p>The gradients multiply the fiber fluxes, and \(\vec g\) is the gradient of the logarithm of the fitted plane at the IFU
      center, so the gradients of terms that multiply add up as vectors there: subtracting \(\vec I\) from \(\vec g\) is
      dividing the measured illumination by the instrument model, up to the curvature of their product (second order, under
      0.04% for these gradients). The plane fitted across the science IFU with the spectrograph factors (pipeline <code>IFUGradient</code>, terms 1, \(y\), \(x\)),
      expressed as the change from the IFU center to its edge, of radius \(\rho\), relative to the mean level \(c_0\).
      <span data-sources="twilight dome">Twilights and dome flats: the coefficients that <code>fit_fiberflat</code> removed from each
      exposure (<code>FIBERFLAT GCOEFF</code> of the per-exposure flats).</span>
      <span data-sources="science">Science frames: the median over the reliable sky lines and continuum windows of the flat field QA of
      each lvmFrame.</span> Strength and direction are the length and angle of \(\vec g_e\).</p>
      <div class="eq">\[ \vec g_e = \frac{\rho}{c_0}\left(c_x,\ c_y\right), \qquad |\vec g_e|, \qquad \theta_e = \operatorname{atan2}\left(g_{e,y},\ g_{e,x}\right) \]</div>
    </div>
    <div class="def">
      <h3>Gradient model</h3>
      <p>An instrument gradient \(\vec I_k\) shared by the exposures of segment \(k\), seen through a rotation by the K-mirror angle
      \(\varphi_e\) when it is fixed before the K-mirror (\(s=\pm1\); \(s=0\) for one fixed in the IFU, no term for none), with an optional
      linear drift \(\vec D_k\) around the mean time \(\bar t_k\) of the segment, plus sky gradients along the unit vectors
      \(\hat u_j\) of the sky directions of each exposure, with amplitudes \(a_j\). The model is linear in all its parameters.</p>
      <div class="eq">\[ \begin{gathered} \vec g_e = R(s\,\varphi_e)\left[\vec I_{k(e)} + \vec D_{k(e)}\,(t_e - \bar t_k)\right] + \sum_j a_j(e)\, \hat u_j(e) + \vec\varepsilon_e \\ R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix} \end{gathered} \]</div>
    </div>
    <div class="def">
      <h3>K-mirror rotation</h3>
      <p>The K-mirror rotates the image by \(\mathrm{PA} - s_F\,\mathrm{FA} = 2\,s_K\,\mathrm{KM} + \mathrm{offset}\), with PA the position
      angle of the science IFU, FA the siderostat field angle and the signs \(s_F, s_K\) of the K-mirror relation calibrated on guider-solved
      frames (\(s_F=-1\) at LCO). The constant offset is absorbed in \(\vec I_k\). <strong>K-m.</strong> is the standard deviation of
      \(\varphi_e\) within a segment: near 0 (a parked K-mirror<span data-sources="twilight dome">, as in twilights and dome flats</span>) the
      K-mirror frames are the IFU frame.</p>
      <div class="eq">\[ \varphi_e = \mathrm{PA}_e - s_F\,\mathrm{FA}_e, \qquad \text{K-m.} = \operatorname{std}_{e \in k}\left(\varphi_e\right) \]</div>
    </div>
    <div class="def">
      <h3>Sky directions</h3>
      <p>The science IFU fibers are placed on the sky with the pipeline astrometry at the middle of the exposure. The direction towards
      the Sun (or the Moon) is the one in which the angular distance \(d\) to it decreases fastest across the IFU, and the direction towards
      the horizon the one in which the altitude \(h\) of the fibers decreases fastest; both from a plane fitted to those values over the fibers.
      <span data-sources="twilight">Twilights use the Sun.</span> <span data-sources="dome">Dome flats use the horizon (the dome
      screen).</span> <span data-sources="science">Science frames use the horizon (the airglow).</span></p>
      <div class="eq">\[ \begin{gathered} \hat u_\odot \parallel -\nabla_{x,y}\, d_\odot(x,y), \qquad \hat u_h \parallel -\nabla_{x,y}\, h(x,y) \\ \hat u_{\mathrm{moon}} \parallel -\nabla_{x,y}\, d_{\mathrm{moon}}(x,y) \end{gathered} \]</div>
    </div>
    <div class="def">
      <h3>Sky amplitudes</h3>
      <p>How the amplitude of each sky gradient varies. <strong>constant</strong>: one value. <strong>depth</strong>: linear in the
      depression of the Sun below \(-3^\circ\) (the twilight sky changes as the Sun sets). <strong>night</strong>: free per night.
      <strong>exposure</strong>: free per exposure; it only removes the component of \(\vec g_e\) along \(\hat u_j\), so the instrument
      gradient is constrained by the perpendicular components, which change as the sky direction moves across the IFU.</p>
      <div class="eq">\[ a = a_0, \qquad a = a_0 + a_1\left(-3^\circ - h_\odot\right), \qquad a = a_\nu, \qquad a = a_e \]</div>
    </div>
    <div class="def">
      <h3>Segments and breaks</h3>
      <p>The instrument gradient may jump at the breaks: the MJDs given, or with <code>--interventions</code> the calibration epochs
      triggered by instrument interventions and warm-ups. Each segment runs from one break to the next and is labeled by the first and last
      night with data. <strong>Nights / exp.</strong> counts the nights and exposures in it. The night-by-night fit uses one segment per night.</p>
      <div class="eq">\[ k(e) = \#\left\{\,b \in \text{breaks} : b \le \mathrm{MJD}_e\,\right\} \]</div>
    </div>
    <div class="def">
      <h3>Fit and residual rms</h3>
      <p>All the exposures of a channel are fitted at once by linear least squares on the \(N = 2n\) components of the \(n\) gradients.
      The parameters that act on one night only (sky amplitudes per exposure or per night, the instrument gradient of a one-night segment)
      are projected out night by night, which gives the same solution. The fit residual \(\vec r_e\) is what neither I nor the
      sky explains; <strong>rms</strong> is its size per exposure, adding the two components; compare it with the spread of the
      gradients themselves. The plot under the direction table gives \(|\vec r_e|\) of each exposure against the night.</p>
      <div class="eq">\[ \vec r_e = \vec g_e - \vec m^{I}_e - \vec m^{\mathrm{sky}}_e, \qquad \mathrm{rms} = \sqrt{\frac{1}{n}\sum_e \left|\vec r_e\right|^2}, \qquad \mathrm{RSS} = \sum_e \left|\vec r_e\right|^2 \]</div>
    </div>
    <div class="def">
      <h3>BIC and &Delta;BIC</h3>
      <p>Bayesian information criterion of each model, with \(k\) parameters (column <strong>k</strong>, counting every sky amplitude).
      It rewards a smaller residual and penalizes parameters; lower is better. &Delta;BIC is the difference from the best model of the
      table: above about 10 the worse model is strongly disfavored. The tiles give &Delta;BIC of the model without an instrument gradient
      relative to the best one with it.</p>
      <div class="eq">\[ \mathrm{BIC} = N \ln\frac{\mathrm{RSS}}{N} + k \ln N, \qquad \Delta\mathrm{BIC} = \mathrm{BIC} - \min_{\text{models}} \mathrm{BIC} \]</div>
    </div>
    <div class="def">
      <h3>CV rms</h3>
      <p>Leave-one-night-out cross validation: each night is predicted with the shared parameters fitted without it, after fitting only its
      own sky amplitudes. Close to the rms, the model generalizes; well above it, the model overfits or the nights disagree.</p>
      <div class="eq">\[ \mathrm{CV\ rms} = \sqrt{\frac{1}{n}\sum_\nu \sum_{e \in \nu} \left|\vec g_e - \vec m_e^{(-\nu)}\right|^2} \]</div>
    </div>
    <div class="def">
      <h3>Errors of I</h3>
      <p>Jackknife over nights: the scatter of the instrument gradient fitted without each of the \(M\) nights, which accounts for the
      errors shared by the exposures of a night. With fewer than 3 nights in a segment the formal least squares error is shown instead,
      marked with *; it assumes independent exposures and underestimates the error. Strength and direction errors propagate those of
      \(I_x, I_y\).</p>
      <div class="eq">\[ \begin{gathered} \sigma^2_{\mathrm{jk}} = \frac{M-1}{M}\sum_\nu \left(\hat I^{(-\nu)} - \bar I\right)^2 \\ \sigma^2_{\mathrm{formal}} = \frac{\mathrm{RSS}}{N - \mathrm{rank}}\left[(A^\top A)^{-1}\right]_{jj} \\ \sigma_{|I|} = \frac{\sqrt{I_x^2\sigma_x^2 + I_y^2\sigma_y^2}}{|I|}, \qquad \sigma_\theta = \frac{\sqrt{I_y^2\sigma_x^2 + I_x^2\sigma_y^2}}{|I|^2} \end{gathered} \]</div>
    </div>
    <div class="def">
      <h3>Degenerate models</h3>
      <p>A model is flagged and never chosen as the best when its parameters are not all determined: the design matrix \(A\) is rank
      deficient, an instrument gradient is less constrained than a single exposure (e.g. a one-night segment with a free sky amplitude per
      night), or a K-mirror frame is fitted where the K-mirror does not move (then it equals the IFU frame, with the same rms and BIC).</p>
      <div class="eq">\[ \mathrm{rank}(A) < k \quad\text{or}\quad \max_k \sigma_{\mathrm{formal}}(I_k) > \tfrac12\,\mathrm{rms} \quad\text{or}\quad \text{K-m.} < 1^\circ \]</div>
    </div>
    <div class="def">
      <h3>Direction of g and g &minus; I</h3>
      <p>Mean resultant length \(R\) and circular mean of the directions of the measured gradient (g) and of the gradient minus the
      instrument model (g &minus; I), in the IFU and relative to each sky direction. \(R=1\) for a fixed direction, near 0 for random ones;
      p is the Rayleigh test of uniform directions. After a correct instrument gradient is removed, what is left is the sky gradient, so R
      relative to its sky direction should rise, with a mean near 0&deg;, and R in the IFU should drop.</p>
      <div class="eq">\[ \begin{gathered} R = \frac{1}{n}\left|\sum_e e^{\,i(\theta_e - \theta^{\mathrm{ref}}_e)}\right|, \qquad \bar\theta = \arg\sum_e e^{\,i(\theta_e - \theta^{\mathrm{ref}}_e)} \\ p \simeq \exp\!\left(\sqrt{1 + 4n + 4(n^2 - n^2R^2)} - (1 + 2n)\right) \end{gathered} \]</div>
    </div>
    <div class="def">
      <h3>Direction rose</h3>
      <p>For each sky direction, the distribution of the angle from that direction to the measured gradient (g) and to the
      gradient minus the instrument model (g &minus; I), as the fraction of exposures per 20&deg; bin; the Moon only for exposures
      with the Moon above the horizon. It shows the same as the R and mean of the table, as a distribution: a sky gradient
      left after removing I piles up near 0&deg; for its sky direction.</p>
      <div class="eq">\[ \Delta\theta_e = \theta(\vec g_e - \vec m^{I}_e) - \theta^{\mathrm{ref}}_e, \qquad \text{fraction per bin} = \frac{\#\{e : \Delta\theta_e \in \text{bin}\}}{n} \]</div>
    </div>
    <div class="def">
      <h3>Maps across the IFU</h3>
      <p>In the browser by calibration epoch, the planes of the measured gradient, of the instrument model at each exposure and of
      the measured illumination divided by the instrument model, evaluated at the science IFU fibers with the positions
      \((x, y)\) scaled to the IFU edge, in % relative to the mean. They are planes because each exposure only keeps the plane
      that the pipeline fitted, not the fluxes of the fibers. The arrow and caption of the last map give
      \(\vec g_e - \vec m^{I}_e\), the gradient of that ratio at the center.</p>
      <div class="eq">\[ \begin{gathered} G(x, y) = 1 + \vec g \cdot \frac{(x, y)}{\rho}, \qquad \text{maps: } G_{\vec g_e} - 1,\quad G_{\vec m^{I}_e} - 1,\quad \frac{G_{\vec g_e}}{G_{\vec m^{I}_e}} - 1 \\ \vec m^{I}_e = R(s\,\varphi_e)\,\vec I_{k(e)}, \qquad \nabla \ln \frac{G_{\vec g_e}}{G_{\vec m^{I}_e}}\Big|_{0} = \vec g_e - \vec m^{I}_e \end{gathered} \]</div>
    </div>
    <div class="def">
      <h3>Geometry of the sky gradient</h3>
      <p>Twilights only. The best model is fitted again with the direction of the Sun rotated by an offset \(\delta\), and mirrored, with
      (sky + I) and without (sky only) the instrument gradient. A minimum near \(\delta = 0\) unmirrored confirms the orientation of the
      geometry; an instrument gradient that only made up for a wrong geometry would not be needed at the best offset, so sky only would get
      close to sky + I there.</p>
      <div class="eq">\[ \theta_\odot \ \to\ \theta_\odot + \delta \quad \text{(direct)}, \qquad \theta_\odot \ \to\ -\theta_\odot + \delta \quad \text{(mirrored)} \]</div>
    </div>
    <div class="def">
      <h3>Twilight I applied to other sources</h3>
      <p>The instrument gradient of the twilights is evaluated at each dome or science exposure (its segment and K-mirror angle) and
      subtracted. The rms of the gradients before and after, and of their sky-only fit before and after, tell how much of the gradient of
      the other source the twilight instrument gradient explains; the table gives the direction of the corrected gradients, which should
      follow the horizon.</p>
      <div class="eq">\[ \vec g'_e = \vec g_e - R(s\,\varphi_e)\,\vec I^{\,\mathrm{twilight}}_{k(e)}, \qquad \sqrt{\tfrac1n \textstyle\sum_e |\vec g_e|^2} \ \to\ \sqrt{\tfrac1n \textstyle\sum_e |\vec g'_e|^2} \]</div>
    </div>
    <div class="def">
      <h3>Trends along the slits</h3>
      <p>The flat field is undone (FLUX &times; SUPERFLAT of the lvmTFlat, lvmDFlat or lvmFrame), so the counts \(c_f\) of each fiber
      \(f\) are the illumination times its throughput: in continuum windows of the twilight and dome flats, and in sky lines of the
      science frames, whose airglow is the same across the science IFU whatever the target. Their logarithm is fitted with a level
      \(\ell\) per telescope and spectrograph, a plane across the science IFU and smooth trends along each slit: a step \(c_0\) for the
      fibers on the upper half of the detector \(U\) and Legendre polynomials of the position \(p_f \in [-1, 1]\) along the slit. The sky
      fibers share the slits and detectors but not the science IFU, so they tell the trends from the plane. The fit without the
      trends gives the plane that the pipeline fits, \(\vec g\); with them, the IFU part \(\vec g_{\mathrm{IFU}}\); their difference is
      the part of \(\vec g\) made by the trends, \(\vec g_{\mathrm{slit}}\). Fibers off by more than 4 robust standard deviations are
      clipped. The table gives the medians of the components over the exposures of each source and segment.</p>
      <div class="eq">\[ \begin{gathered} \ln c_f = \ell_{t(f),s(f)} + \Big[c_{0,s}\,[f \in U] + \sum_{k=1}^{3} c_{k,s}\,P_k(p_f)\Big] + [f \in \mathrm{Sci}]\ \vec\gamma\cdot(\vec x_f - \bar{\vec x}) \\ \vec g = \rho\,\vec\gamma_{\text{no trends}}, \qquad \vec g_{\mathrm{IFU}} = \rho\,\vec\gamma, \qquad \vec g_{\mathrm{slit}} = \vec g - \vec g_{\mathrm{IFU}} \end{gathered} \]</div>
    </div>
  </div>
</section>"""

# Sun altitude (degrees) where the twilight sky amplitude of the 'depth' model is referred to
SUN_ALT_REF = -3.0


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def intervention_breaks(epochs: Dict[int, Dict] = None, triggers: Sequence[str] = INTERVENTION_TRIGGERS) -> List[int]:
    """MJDs of the calibration epochs whose trigger may have changed the instrument

    Parameters
    ----------
    epochs : dict, optional
        Calibration epochs, as from :func:`lvmdrp.qa.fiberflats.load_calibration_epochs`.
        Default is the epochs file in lvmcore.
    triggers : list[str], optional
        Triggers, case insensitive, that start a new segment. Default is
        :data:`INTERVENTION_TRIGGERS`.

    Returns
    -------
    list[int]
        Sorted MJDs.
    """
    epochs = epochs if epochs is not None else load_calibration_epochs()
    triggers = [t.lower() for t in triggers]
    return sorted(int(mjd) for mjd, epoch in epochs.items() if str(epoch.get("trigger", "")).lower() in triggers)


def _kmirror_rotation(data: pd.DataFrame, field_sign: int) -> np.ndarray:
    """Rotation of the image by the K-mirror up to a constant: PA - field_sign * field angle, in degrees"""
    if "field" not in data.columns:
        data["field"] = np.nan
        good = data[["ra", "dec"]].notna().all(axis=1) & data["time"].notna() if {"ra", "dec", "time"} <= set(data.columns) else []
        if np.any(good):
            data.loc[good, "field"] = siderostat_field_angle(data.loc[good, "ra"].to_numpy(float), data.loc[good, "dec"].to_numpy(float),
                                                             Time(list(data.loc[good, "time"]), scale="utc"))
    pa = data["pa"].astype(float) if "pa" in data.columns else np.nan
    return _wrap_angle(pa - field_sign * data["field"].astype(float))


def prepare_gradients(data: Union[str, pd.DataFrame], source: str, kmirror=None, channels: str = "brz") -> pd.DataFrame:
    """Gradients of one source with the columns the fits need

    Parameters
    ----------
    data : str or pandas.DataFrame
        Table of a trends dashboard (twilight or dome) or from
        :func:`science_gradients`, or the path of its CSV file. Needs ``expnum``,
        ``mjd`` (night), ``channel``, ``grad_x`` and ``grad_y`` (fractions), and
        the direction columns of the sky model (e.g. ``dir_sun``).
    source : str
        'twilight', 'dome' or 'science'.
    kmirror : dict or str, optional
        K-mirror relation (see :func:`lvmdrp.qa.fiberflats.load_kmirror_relation`),
        for the sign of the field angle in the K-mirror rotation. Default is the
        sign found at LCO (-1).
    channels : str, optional
        Channels to keep. Default is 'brz'.

    Returns
    -------
    pandas.DataFrame
        The rows with a gradient, plus ``source``, ``gx`` and ``gy`` (%),
        ``depth`` (degrees of the Sun below -3 deg), ``field`` (siderostat
        field angle) and ``phi`` (rotation of the K-mirror up to a constant,
        PA - field_sign x field angle).
    """
    data = pd.read_csv(data) if isinstance(data, (str, os.PathLike)) else data.copy()
    data = data[data["channel"].astype(str).isin(list(channels))]
    data = data[np.isfinite(data["grad_x"].astype(float)) & np.isfinite(data["grad_y"].astype(float))].reset_index(drop=True)
    data["source"] = source
    data["gx"], data["gy"] = 100 * data["grad_x"].astype(float), 100 * data["grad_y"].astype(float)
    data["depth"] = SUN_ALT_REF - data["sun_alt"].astype(float) if "sun_alt" in data.columns else np.nan
    relation = load_kmirror_relation(kmirror)
    data["phi"] = _kmirror_rotation(data, relation["field_sign"] if relation else -1)
    return data


def science_gradients(summary: Union[str, Table], drpver: str = None, redux_dir: str = None, kmirror=None) -> pd.DataFrame:
    """Gradient across the science IFU of each science frame, with its sky geometry

    The gradients are the per frame medians of the sky line and continuum
    measurements of the flat field QA (:func:`lvmdrp.qa.fiberflats.frame_gradients`);
    the geometry (directions of the horizon, the Moon and the Sun across the
    IFU, position angle and siderostat field angle) comes from the header of
    the lvmFrame with :func:`lvmdrp.qa.fiberflats.twilight_geometry`.

    Parameters
    ----------
    summary : str or astropy.table.Table
        Summary table of the flat field QA, or the path of its FITS file.
    drpver : str, optional
        Pipeline version of the lvmFrames. Default is the one in the summary.
    redux_dir : str, optional
        Reductions root directory. Default is $LVM_SPECTRO_REDUX.
    kmirror : dict or str, optional
        K-mirror relation for the frames without ``POSCIPA``.

    Returns
    -------
    pandas.DataFrame
        One row per frame and channel, with the columns of a trends table.
    """
    from lvmdrp.main import read_fibermap

    summary = Table.read(summary, hdu=1) if isinstance(summary, (str, os.PathLike)) else Table(summary, copy=True)
    # the string columns come back from FITS as bytes
    summary.convert_bytestring_to_unicode()
    gradients = frame_gradients(summary).to_pandas()
    if gradients.empty:
        raise ValueError("no reliable gradient measurements in the flat field QA summary")
    redux_dir = redux_dir or os.getenv("LVM_SPECTRO_REDUX", ".")
    frames = summary.to_pandas()[["filename", "tileid", "mjd"] + (["drpver"] if "drpver" in summary.colnames else [])].drop_duplicates("filename")
    gradients = gradients.merge(frames.drop(columns="mjd"), on="filename", how="left")

    relation = load_kmirror_relation(kmirror)
    fibermap = read_fibermap(as_table=True)
    geometry = {}
    for row in gradients.drop_duplicates("expnum").itertuples():
        version = drpver or getattr(row, "drpver", None)
        path = os.path.join(redux_dir, str(version), tileid_grp(int(row.tileid)), str(int(row.tileid)), str(int(row.mjd)), row.filename)
        if not os.path.isfile(path):
            log.warning(f"missing {path}, skipping the geometry of exposure {row.expnum}")
            continue
        header = fits.getheader(path, 0)
        geometry[int(row.expnum)] = {"expnum": int(row.expnum), "mjd_night": _night_sjd(header, path),
                                     **twilight_geometry(header, fibermap, kmirror=relation)}
    if not geometry:
        raise FileNotFoundError(f"none of the lvmFrames of the summary are in {redux_dir}")
    data = gradients.merge(pd.DataFrame(list(geometry.values())), on="expnum", how="inner")
    data["mjd"] = data.pop("mjd_night")
    data["strength"] = np.hypot(data["grad_x"], data["grad_y"])
    data["direction"] = np.degrees(np.arctan2(data["grad_y"], data["grad_x"]))
    return data


# continuum windows of the twilight and dome flats, and sky lines of the science frames, where the slit trends are measured
SLIT_WINDOWS = {"b": ((4650, 4750), (5300, 5400)), "r": ((6650, 6750), (7300, 7400)), "z": ((8600, 8700), (9150, 9250))}
# sky lines away from the bright nebular lines ([OII] 7320/7330, [SII], Paschen), so the science fibers see only the airglow
SLIT_SKY_LINES = {"b": (5577.34,), "r": (6863.96, 7276.41), "z": (8885.85, 9375.98)}

# source of each product the slit trends are measured on, and whether in the continuum or in the sky lines
SLIT_PRODUCTS = {"lvmTFlat": ("twilight", "continuum"), "lvmDFlat": ("dome", "continuum"), "lvmFrame": ("science", "lines")}
# header values kept with the split, to look for what drives the IFU part: column, keyword
SLIT_HEADER_FLOATS = {"exptime": "EXPTIME", "labtemp": "LABTEMP", "labhumid": "LABHUMID", "tempsci": "TEMPSCI",
                      "kmirror": "TESCIKM", "focus": "TESCIFO", "airmass": "TESCIAM"}


def _line_counts(flux: np.ndarray, wave: np.ndarray, cwave: float, half_width: float = 6.0, gap: float = 10.0, width: float = 10.0) -> np.ndarray:
    """Flux of a sky line in each fiber, over the median of the continuum on both sides"""
    core = np.abs(wave - cwave) <= half_width
    side = (np.abs(wave - cwave) > gap) & (np.abs(wave - cwave) <= gap + width)
    continuum = np.nanmedian(np.where(side, flux, np.nan), axis=1)
    return np.nansum(np.where(core, flux - continuum[:, None], 0.0), axis=1) / np.maximum(core.sum(axis=1), 1)


def slit_trend_split(paths: Sequence[str], degree: int = 3, kmirror=None) -> pd.DataFrame:
    """Split of the gradient across the science IFU into the trends along the slits and an IFU part, from lvmTFlat, lvmDFlat or lvmFrame files

    The flat field is undone (FLUX x SUPERFLAT), so the counts of each fiber
    are the illumination times the fiber throughput. In the continuum of the
    twilight and dome flats, and in sky lines of the science frames (the
    airglow is the same across the science IFU, whatever the target), the
    counts are fitted with :func:`lvmdrp.core.rss.fit_slit_trend_model`: a level
    per telescope and spectrograph, smooth trends along each slit (a step for
    the upper half of the detector and Legendre polynomials) measured with the
    sky fibers, and a plane across the science IFU. The same fit without the
    trends gives the plane that the pipeline fits (g); with them, the IFU part;
    the difference is the part of g that the trends along the slits make.

    Parameters
    ----------
    paths : list[str]
        Paths of the files, named lvmTFlat-*, lvmDFlat-* or lvmFrame-*.
    degree : int, optional
        Degree of the Legendre polynomials along the slits. Default is 3.
    kmirror : dict or str, optional
        K-mirror relation, for the sky geometry of the frames without ``POSCIPA``.

    Returns
    -------
    pandas.DataFrame
        One row per file and window or sky line: ``source``, ``measure``
        ('continuum' or 'lines'), ``window``, ``filename``, ``expnum``, ``mjd``,
        ``channel``, the vectors at the IFU edge in % (``all_x/y``: the
        pipeline plane g, ``slit_x/y``: the part of the trends along the slits,
        ``ifu_x/y``: the rest), the log coefficients of the trends
        (``c{spectrograph}_{k}``, k=0 the step, 1.. the Legendre terms),
        ``nfit`` (fibers kept), ``obstime``, ``tileid``, the header values of
        ``SLIT_HEADER_FLOATS`` (NaN if missing) and the sky geometry of
        :func:`lvmdrp.qa.fiberflats.twilight_geometry`.
    """
    from lvmdrp.core.rss import fit_slit_trend_model
    from lvmdrp.core.tracemask import TraceMask

    relation = load_kmirror_relation(kmirror)
    rows = []
    for path in paths:
        name = os.path.basename(path)
        product = name.split("-")[0]
        if product not in SLIT_PRODUCTS:
            log.warning(f"{name} is not an lvmTFlat, lvmDFlat or lvmFrame, skipping")
            continue
        source, measure = SLIT_PRODUCTS[product]
        with fits.open(path) as hdus:
            header = hdus[0].header
            flux = hdus["FLUX"].data.astype(float)
            if "SUPERFLAT" in hdus:
                flux = flux * hdus["SUPERFLAT"].data
            flux[hdus["MASK"].data.astype(bool)] = np.nan
            wave = TraceMask.from_coeff_table(Table(hdus["WAVE_TRACE"].data))._data
            cent = TraceMask.from_coeff_table(Table(hdus["CENT_TRACE"].data))._data
            slitmap = Table(hdus["SLITMAP"].data)
        channel = name.split("-")[1][0]
        telescope = np.array([str(t).strip() for t in slitmap["telescope"]])
        x, y = np.asarray(slitmap["xpmm"], dtype=float), np.asarray(slitmap["ypmm"], dtype=float)
        science = telescope == "Sci"
        rho = np.max(np.hypot(x[science] - x[science].mean(), y[science] - y[science].mean()))
        try:
            geometry = twilight_geometry(header, slitmap, kmirror=relation)
        except Exception as e:
            log.warning(f"cannot compute the sky geometry of {name} ({type(e).__name__}: {e})")
            geometry = {}
        windows = SLIT_WINDOWS[channel] if measure == "continuum" else [(w, w) for w in SLIT_SKY_LINES[channel]]
        for lo, hi in windows:
            if measure == "continuum":
                counts = np.nanmedian(np.where((wave >= lo) & (wave < hi), flux, np.nan), axis=1)
                label = f"{lo:.0f}–{hi:.0f} Å"
            else:
                counts = _line_counts(flux, wave, lo)
                label = f"{lo:.1f} Å"
            column = np.argmin(np.abs(wave - 0.5 * (lo + hi)), axis=1)
            upper = cent[np.arange(len(cent)), column] >= 2040
            plain = fit_slit_trend_model(counts, slitmap, upper=upper, degree=degree, with_trends=False)
            model = fit_slit_trend_model(counts, slitmap, upper=upper, degree=degree)
            g_all, g_ifu = 100 * rho * plain["plane"], 100 * rho * model["plane"]
            row = {"source": source, "measure": measure, "window": label, "filename": name, "expnum": int(header.get("EXPOSURE", 0)),
                   "mjd": _night_sjd(header, path), "channel": channel, "all_x": g_all[0], "all_y": g_all[1],
                   "slit_x": g_all[0] - g_ifu[0], "slit_y": g_all[1] - g_ifu[1], "ifu_x": g_ifu[0], "ifu_y": g_ifu[1],
                   "nfit": int(model["fitted"].sum()), "obstime": str(header.get("OBSTIME", "")), "tileid": header.get("TILE_ID"),
                   **{column: _header_float(header, key) for column, key in SLIT_HEADER_FLOATS.items()}}
            for i, coeffs in enumerate(model["coeffs"]):
                row.update({f"c{i + 1}_{k}": float(c) for k, c in enumerate(coeffs)})
            rows.append({**row, **geometry})
        log.info(f"slit trends of {name}: {len(windows)} {'windows' if measure == 'continuum' else 'sky lines'}")
    data = pd.DataFrame(rows)
    for part in ("all", "slit", "ifu"):
        if len(data):
            data[f"{part}_g"] = np.hypot(data[f"{part}_x"], data[f"{part}_y"])
            data[f"{part}_dir"] = np.degrees(np.arctan2(data[f"{part}_y"], data[f"{part}_x"]))
    return data


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _segments(mjd: np.ndarray, breaks) -> np.ndarray:
    """Segment of each night: 0 before the first break, k from the k-th break on; 'nights' gives a segment per night"""
    if isinstance(breaks, str) and breaks == "nights":
        return np.unique(mjd, return_inverse=True)[1]
    return np.searchsorted(np.sort(np.asarray(breaks if breaks is not None else [], dtype=float)), mjd, side="right")


def _segment_edges(mjd: np.ndarray, breaks) -> Dict[int, Tuple[float, float]]:
    """First and last night of the data in each segment"""
    segment = _segments(mjd, breaks)
    return {int(k): (float(mjd[segment == k].min()), float(mjd[segment == k].max())) for k in np.unique(segment)}


def gradient_design(data: pd.DataFrame, frame: str = "ifu", sky: Dict[str, Dict[str, str]] = None, breaks=None,
                    drift: bool = False) -> Tuple[sparse.csc_matrix, np.ndarray, List[Tuple]]:
    """Design matrix of the gradient model, see the module docstring

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel, from :func:`prepare_gradients` (one or more sources).
    frame : str, optional
        Frame of the instrument gradient, a key of :data:`INSTRUMENT_FRAMES`. Default is 'ifu'.
    sky : dict, optional
        Sky gradients of each source, {source: {direction column: amplitude}},
        with the amplitudes in :data:`SKY_AMPLITUDES`. Default is :data:`DEFAULT_SKY`.
    breaks : list[int] or 'nights', optional
        MJDs where the instrument gradient may jump, or 'nights' for one per
        night. Default is none.
    drift : bool, optional
        Add a linear drift of the instrument gradient within each segment. Default is False.

    Returns
    -------
    tuple
        Sparse design matrix (2n x k: the x components of the n exposures,
        then the y components), target vector (2n) and the label of each parameter:
        ('inst', segment, 'x'|'y'), ('drift', segment, 'x'|'y') or
        ('sky', source, column, amplitude, group).
    """
    sky = sky or DEFAULT_SKY
    n = len(data)
    mjd = data["mjd"].to_numpy(float)
    rows, cols, values, labels = [], [], [], []

    def add(index, x, y, label):
        """Column with the values x and y at the x and y components of the exposures in `index`"""
        rows.extend([index, index + n])
        cols.append(np.full(2 * index.size, len(labels)))
        values.extend([x, y])
        labels.append(label)

    rotation = INSTRUMENT_FRAMES[frame][1]
    if rotation is not None:
        phi = np.radians(rotation * data["phi"].to_numpy(float)) if rotation else np.zeros(n)
        cos, sin = np.cos(phi), np.sin(phi)
        segment = _segments(mjd, breaks)
        for k in np.unique(segment):
            index = np.flatnonzero(segment == k)
            # R(phi) (Ix, Iy) = (cos Ix - sin Iy, sin Ix + cos Iy)
            add(index, cos[index], sin[index], ("inst", int(k), "x"))
            add(index, -sin[index], cos[index], ("inst", int(k), "y"))
            if drift:
                years = (mjd[index] - mjd[index].mean()) / 365.25
                add(index, years * cos[index], years * sin[index], ("drift", int(k), "x"))
                add(index, -years * sin[index], years * cos[index], ("drift", int(k), "y"))

    for source, terms in sky.items():
        in_source = (data["source"] == source).to_numpy()
        if not in_source.any():
            continue
        for column, amplitude in terms.items():
            if amplitude not in SKY_AMPLITUDES:
                raise ValueError(f"unknown sky amplitude {amplitude!r}, use one of {SKY_AMPLITUDES}")
            angle = np.radians(data[column].to_numpy(float))
            ux, uy = np.cos(angle), np.sin(angle)
            if amplitude in ("constant", "depth"):
                groups = {"all": np.flatnonzero(in_source)}
            else:
                key = data["mjd" if amplitude == "night" else "expnum"].to_numpy()
                order = np.flatnonzero(in_source)
                values_in, inverse = np.unique(key[order], return_inverse=True)
                groups = {value: order[inverse == i] for i, value in enumerate(values_in)}
            for group, index in groups.items():
                add(index, ux[index], uy[index], ("sky", source, column, amplitude, group))
            if amplitude == "depth":
                index = groups["all"]
                depth = np.nan_to_num(data["depth"].to_numpy(float))[index]
                add(index, depth * ux[index], depth * uy[index], ("sky", source, column, "slope", "all"))
    target = np.r_[data["gx"].to_numpy(float), data["gy"].to_numpy(float)]
    if labels:
        design = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, len(labels)))
    else:
        design = sparse.csc_matrix((2 * n, 0))
    return design, target, labels


def _usable(data: pd.DataFrame, frame: str, sky: Dict[str, Dict[str, str]]) -> np.ndarray:
    """Rows with all the values the model needs"""
    good = np.isfinite(data["gx"].to_numpy(float)) & np.isfinite(data["gy"].to_numpy(float))
    if INSTRUMENT_FRAMES[frame][1]:
        good &= np.isfinite(data["phi"].to_numpy(float))
    for source, terms in sky.items():
        in_source = (data["source"] == source).to_numpy()
        for column, amplitude in terms.items():
            values = data[column].to_numpy(float) if column in data.columns else np.full(len(data), np.nan)
            ok = np.isfinite(values)
            if amplitude == "depth":
                ok &= np.isfinite(data["depth"].to_numpy(float))
            good &= ~in_source | ok
    return good


def _single_thread():
    """Limit BLAS to one thread: the many small least squares fits are much slower with multithreaded OpenBLAS"""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _solve(design, target):
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return params, target - design @ params


def _eliminate(design: sparse.csc_matrix, target: np.ndarray, nights: np.ndarray) -> Dict:
    """Projects out the parameters that only act on one night, leaving a small least squares problem for the shared ones

    The sky amplitudes per exposure or per night and the instrument gradient
    of a single night only enter the equations of their night. Within each
    night, the equations are projected onto the complement of the columns of
    those local parameters; the shared parameters fitted to the projected
    equations are the least squares solution of the whole problem, and the
    local parameters follow night by night. This keeps the cost linear in the
    number of exposures.

    Returns
    -------
    dict
        shared (indices of the shared columns), local ({night: (rows, columns,
        block)}), projected design (dense, shared columns) and target, and
        local_rank (summed rank of the local blocks).
    """
    rows_night = np.r_[nights, nights]
    k = design.shape[1]
    shared = np.ones(k, dtype=bool)
    local = {}
    for j in range(k):
        rows = design.indices[design.indptr[j]:design.indptr[j + 1]]
        if rows.size:
            owners = np.unique(rows_night[rows])
            if owners.size == 1:
                local.setdefault(owners[0], []).append(j)
                shared[j] = False
    shared = np.flatnonzero(shared)
    projected = design[:, shared].toarray()
    b = target.copy()
    by_rows = design.tocsr()
    blocks, local_rank = {}, 0
    for night, columns in local.items():
        rows = np.flatnonzero(rows_night == night)
        block = by_rows[rows][:, columns].toarray()
        u, singular, _ = np.linalg.svd(block, full_matrices=False)
        basis = u[:, singular > singular.max() * 1e-10] if singular.size and singular.max() > 0 else u[:, :0]
        local_rank += basis.shape[1]
        projected[rows] -= basis @ (basis.T @ projected[rows])
        b[rows] -= basis @ (basis.T @ b[rows])
        blocks[night] = (rows, np.asarray(columns), block)
    return {"shared": shared, "local": blocks, "design": projected, "target": b, "local_rank": local_rank}


def _solve_eliminated(design, target, reduced) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Parameters, residuals, rank and unscaled formal variances of the problem reduced by :func:`_eliminate`"""
    k = design.shape[1]
    params = np.zeros(k)
    variance = np.full(k, np.nan)
    shared = reduced["shared"]
    rank = reduced["local_rank"]
    if shared.size:
        params[shared], _ = _solve(reduced["design"], reduced["target"])
        rank += int(np.linalg.matrix_rank(reduced["design"]))
        variance[shared] = np.diag(np.linalg.pinv(reduced["design"].T @ reduced["design"]))
    shared_model = design[:, shared] @ params[shared] if shared.size else np.zeros(target.size)
    for rows, columns, block in reduced["local"].values():
        params[columns], _ = _solve(block, target[rows] - shared_model[rows])
        # conditional on the shared parameters
        variance[columns] = np.diag(np.linalg.pinv(block.T @ block))
    return params, target - design @ params, rank, variance


def _holdout_residuals(reduced, nights, k):
    """Leave-one-night-out residuals and the shared parameters fitted without each night

    The parameters local to the held-out night (its own sky amplitudes, or the
    instrument gradient of a segment with that night only) are already
    projected out of its equations, and shared parameters that only the
    held-out night constrains are fitted on that night with the others fixed,
    so the residuals measure how well the other nights predict the shared
    parameters.
    """
    design, target, shared = reduced["design"], reduced["target"], reduced["shared"]
    rows_night = np.r_[nights, nights]
    residuals = np.full(target.shape, np.nan)
    jackknife = []
    for night in np.unique(nights):
        test = rows_night == night
        train = ~test
        params = np.full(k, np.nan)
        if shared.size:
            supported = np.any(design[train] != 0, axis=0)
            fitted = np.full(shared.size, np.nan)
            fitted[supported], _ = _solve(design[train][:, supported], target[train])
            predicted = design[test][:, supported] @ fitted[supported]
            if (~supported).any():
                extra, _ = _solve(design[test][:, ~supported], target[test] - predicted)
                predicted = predicted + design[test][:, ~supported] @ extra
            params[shared] = fitted
        else:
            predicted = np.zeros(test.sum())
        residuals[test] = target[test] - predicted
        jackknife.append(params)
    return residuals, np.array(jackknife)


def fit_gradient_model(data: pd.DataFrame, frame: str = "ifu", sky: Dict[str, Dict[str, str]] = None, breaks=None,
                       drift: bool = False, crossval: bool = True) -> Dict:
    """Fits the gradient model of the module docstring to the gradients of one channel

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel, from :func:`prepare_gradients` (one or more
        sources; the instrument gradient is shared by all of them).
    frame, sky, breaks, drift
        See :func:`gradient_design`.
    crossval : bool, optional
        Compute the leave-one-night-out rms and the jackknife errors of the
        instrument gradient. Default is True.

    Returns
    -------
    dict
        n (exposures), nights, k (parameters), rank, rms (% per exposure, of
        the 2D residual), cv_rms (leave-one-night-out), bic, frame, breaks,
        instrument (pandas.DataFrame per segment: first and last night, n,
        gx, gy, their jackknife errors over nights (NaN with fewer than 3
        nights) and formal errors, strength, direction, spread of the K-mirror
        angle), drift (likewise, %/yr), sky (pandas.DataFrame of the
        sky amplitudes, with jackknife and formal errors), and data (the rows used, with the instrument and sky
        model and the residuals: inst_x, inst_y, sky_x, sky_y, res_x, res_y).
    """
    with _single_thread():
        return _fit_gradient_model(data, frame, sky or DEFAULT_SKY, breaks, drift, crossval)


def _fit_gradient_model(data, frame, sky, breaks, drift, crossval):
    good = _usable(data, frame, sky)
    data = data[good].reset_index(drop=True)
    n = len(data)
    result = {"n": n, "nights": int(data["mjd"].nunique()), "frame": frame,
              "breaks": breaks if isinstance(breaks, str) else list(breaks or [])}
    if n < 3:
        for column in ("inst_x", "inst_y", "sky_x", "sky_y", "res_x", "res_y"):
            data[column] = np.nan
        return {**result, "k": 0, "rank": 0, "rms": np.nan, "cv_rms": np.nan, "bic": np.nan,
                "instrument": pd.DataFrame(), "drift": pd.DataFrame(), "sky": pd.DataFrame(), "data": data}

    design, target, labels = gradient_design(data, frame=frame, sky=sky, breaks=breaks, drift=drift)
    nights = data["mjd"].to_numpy()
    reduced = _eliminate(design, target, nights)
    params, residual, rank, variance = _solve_eliminated(design, target, reduced)
    rss = float(residual @ residual)
    size, k = target.size, design.shape[1]
    result.update({"k": k, "rank": rank,
                   "rms": float(np.sqrt(rss / n)), "bic": float(size * np.log(max(rss, 1e-12) / size) + k * np.log(size))})

    jackknife = None
    if crossval and result["nights"] > 2:
        cv_residual, jackknife = _holdout_residuals(reduced, nights, k)
        result["cv_rms"] = float(np.sqrt(np.nansum(cv_residual ** 2) / n))
    else:
        result["cv_rms"] = np.nan

    # formal errors, which ignore that the exposures of a night share their errors; they show how well each parameter is constrained
    with np.errstate(invalid="ignore"):
        formal = np.sqrt(rss / max(size - rank, 1) * variance)

    def errors(index):
        if jackknife is None:
            return np.nan
        values = jackknife[:, index]
        values = values[np.isfinite(values)]
        m = values.size
        return float(np.sqrt((m - 1) / m * np.sum((values - values.mean()) ** 2))) if m > 2 else np.nan

    mjd = data["mjd"].to_numpy(float)
    segment = _segments(mjd, breaks)
    edges = _segment_edges(mjd, breaks)
    phi = data["phi"].to_numpy(float)

    def per_segment(kind):
        rows = []
        for k_seg in sorted(edges):
            ix = [i for i, label in enumerate(labels) if label[:2] == (kind, k_seg)]
            if len(ix) != 2:
                continue
            gx, gy = params[ix]
            in_segment = segment == k_seg
            nights = int(np.unique(mjd[in_segment]).size)
            # the jackknife over nights needs a few nights in the segment to see the scatter of its own nights
            gx_err, gy_err = (errors(ix[0]), errors(ix[1])) if nights >= 3 else (np.nan, np.nan)
            rows.append({"segment": k_seg, "mjd_first": edges[k_seg][0], "mjd_last": edges[k_seg][1],
                         "n": int(in_segment.sum()), "nights": nights, "gx": float(gx), "gy": float(gy),
                         "gx_err": gx_err, "gy_err": gy_err, "gx_sig": float(formal[ix[0]]), "gy_sig": float(formal[ix[1]]),
                         "strength": float(np.hypot(gx, gy)), "direction": float(np.degrees(np.arctan2(gy, gx))),
                         "phi_spread": float(np.nanstd(phi[in_segment])) if np.isfinite(phi[in_segment]).any() else np.nan})
        return pd.DataFrame(rows)

    result["instrument"] = per_segment("inst")
    result["drift"] = per_segment("drift")
    result["sky"] = pd.DataFrame([{"source": label[1], "direction": label[2], "amplitude": label[3], "group": label[4],
                                   "value": float(params[i]), "error": errors(i) if label[4] == "all" else np.nan, "sigma": float(formal[i])}
                                  for i, label in enumerate(labels) if label[0] == "sky"])

    is_inst = np.array([label[0] in ("inst", "drift") for label in labels], dtype=bool)
    inst = design[:, is_inst] @ params[is_inst] if is_inst.any() else np.zeros(target.size)
    sky_model = design[:, ~is_inst] @ params[~is_inst] if (~is_inst).any() else np.zeros(target.size)
    data["inst_x"], data["inst_y"] = inst[:n], inst[n:]
    data["sky_x"], data["sky_y"] = sky_model[:n], sky_model[n:]
    data["res_x"], data["res_y"] = residual[:n], residual[n:]
    result["data"] = data
    return result


def _label_breaks(breaks) -> str:
    if isinstance(breaks, str):
        return breaks
    return ", ".join(str(int(b)) for b in breaks) if breaks else "none"


def _label_sky(sky: Dict[str, Dict[str, str]], sources) -> str:
    return "; ".join(f"{source}: " + " + ".join(f"{column.replace('dir_', '')} ({amplitude})" for column, amplitude in terms.items())
                     for source, terms in sky.items() if source in sources)


def _instrument_label(instrument: pd.DataFrame) -> str:
    if instrument.empty:
        return "-"
    return " | ".join(f"{row.strength:.2f}% @ {row.direction:.0f}°" for row in instrument.itertuples())


def compare_gradient_models(data: pd.DataFrame, frames: Sequence[str] = tuple(INSTRUMENT_FRAMES), skies: Dict[str, Dict] = None,
                            breaks: Dict[str, object] = None, drift: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Fits every combination of instrument frame, sky model and breaks to the gradients of one channel

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel (one or more sources), from :func:`prepare_gradients`.
    frames : list[str], optional
        Instrument frames, keys of :data:`INSTRUMENT_FRAMES`. Default is all.
    skies : dict, optional
        Sky models to try, {label: {source: {direction column: amplitude}}}.
        Default is :data:`DEFAULT_SKY` only.
    breaks : dict, optional
        Breaks to try, {label: list of MJDs or 'nights'}. Default is no breaks.
    drift : bool, optional
        Also try a linear drift within the segments. Default is False.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        One row per model (frame, sky, breaks, drift, n, k, rank, rms, cv_rms,
        bic, dbic from the lowest BIC, degenerate: rank deficient, an instrument
        gradient with a formal error above half the rms, or a K-mirror frame
        with a fixed K-mirror; instrument), sorted by BIC,
        and the fits by model key (frame, sky label, breaks label, drift).
    """
    skies = skies or {"default": DEFAULT_SKY}
    breaks = breaks or {"none": []}
    sources = set(data["source"])
    rows, fits = [], {}
    for frame in frames:
        for sky_label, sky in skies.items():
            for breaks_label, breaks_value in breaks.items():
                for with_drift in ((False, True) if drift else (False,)):
                    if INSTRUMENT_FRAMES[frame][1] is None and (breaks_label != list(breaks)[0] or with_drift):
                        continue  # without an instrument gradient, the breaks and the drift change nothing
                    fit = fit_gradient_model(data, frame=frame, sky=sky, breaks=breaks_value, drift=with_drift)
                    key = (frame, sky_label, breaks_label, with_drift)
                    fits[key] = fit
                    instrument = fit["instrument"]
                    phi_spread = instrument["phi_spread"].max() if not instrument.empty else np.nan
                    # an instrument gradient less constrained than a single exposure, e.g. one night with a free sky amplitude per night
                    loose = not instrument.empty and bool((instrument[["gx_sig", "gy_sig"]].max(axis=1) > 0.5 * fit["rms"]).any())
                    rows.append({"frame": frame, "sky": sky_label, "sky_model": _label_sky(sky, sources), "breaks": breaks_label,
                                 "drift": with_drift, "n": fit["n"], "nights": fit["nights"], "k": fit["k"], "rank": fit["rank"],
                                 "rms": fit["rms"], "cv_rms": fit["cv_rms"], "bic": fit["bic"],
                                 "degenerate": bool(fit["rank"] < fit["k"] or loose
                                                    or (INSTRUMENT_FRAMES[frame][1] and np.isfinite(phi_spread) and phi_spread < 1.0)),
                                 "instrument": _instrument_label(fit["instrument"])})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["dbic"] = table["bic"] - table["bic"].min()
        table = table.sort_values("bic").reset_index(drop=True)
    return table, fits


def residual_alignment(fit: Dict, references: Sequence[str] = ("dir_sun", "dir_horizon", "dir_moon")) -> pd.DataFrame:
    """Direction of the residual gradients, after the instrument gradient is removed, in the IFU and relative to the sky directions

    After a correct instrument gradient is removed, what is left of the
    measured gradient is the sky gradient, so its direction should stay fixed
    relative to its sky direction (mean resultant length R near 1) and change
    in IFU coordinates.

    Parameters
    ----------
    fit : dict
        Fit from :func:`fit_gradient_model`.
    references : list[str], optional
        Direction columns to compare with. Default is the Sun, the horizon and the Moon.

    Returns
    -------
    pandas.DataFrame
        For the measured gradient (g) and the gradient minus the instrument
        model (g - I): frame, R, circular mean, Rayleigh p-value and n.
    """
    data = fit["data"]
    rows = []
    for label, gx, gy in (("g", data["gx"], data["gy"]), ("g − I", data["gx"] - data["inst_x"], data["gy"] - data["inst_y"])):
        direction = np.degrees(np.arctan2(np.asarray(gy, dtype=float), np.asarray(gx, dtype=float)))
        for frame, reference in [("IFU", None)] + [(ref.replace("dir_", "").capitalize(), ref) for ref in references if ref in data.columns]:
            angles = direction if reference is None else _wrap_angle(direction - data[reference].to_numpy(float))
            mean, length = _resultant_length(angles)
            n = int(np.isfinite(angles).sum())
            rows.append({"gradient": label, "frame": frame, "R": length, "mean": mean, "p": _rayleigh_pvalue(n, length), "n": n})
    return pd.DataFrame(rows)


def sky_direction_scan(data: pd.DataFrame, column: str = "dir_sun", amplitude: str = "depth", breaks=None,
                       offsets: Sequence[float] = tuple(range(-180, 180, 5))) -> pd.DataFrame:
    """Fits with the sky direction rotated and mirrored, to test the geometry of the sky gradient

    An error in the position angle or in the orientation of the IFU would
    shift (or mirror) the computed direction of the sky gradient. With a right
    geometry, the fit is best with no offset; an instrument gradient that only
    makes up for a wrong geometry would be needed less at the best offset.

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel and one source, from :func:`prepare_gradients`.
    column : str, optional
        Direction of the sky gradient. Default is 'dir_sun'.
    amplitude : str, optional
        Its amplitude, see :data:`SKY_AMPLITUDES`. Default is 'depth'.
    breaks : list[int], optional
        Breaks of the instrument gradient. Default is none.
    offsets : list[float], optional
        Offsets added to the direction, in degrees.

    Returns
    -------
    pandas.DataFrame
        offset, mirrored, rms and bic with an IFU-fixed instrument gradient
        (rms_inst, bic_inst) and without one (rms_sky, bic_sky).
    """
    source = data["source"].iloc[0]
    rows = []
    for mirrored in (False, True):
        for offset in offsets:
            shifted = data.copy()
            shifted[column] = (-1 if mirrored else 1) * data[column].astype(float) + offset
            row = {"offset": float(offset), "mirrored": mirrored}
            for frame, suffix in (("ifu", "inst"), ("none", "sky")):
                fit = fit_gradient_model(shifted, frame=frame, sky={source: {column: amplitude}}, breaks=breaks, crossval=False)
                row.update({f"rms_{suffix}": fit["rms"], f"bic_{suffix}": fit["bic"]})
            rows.append(row)
    return pd.DataFrame(rows)


def predict_instrument(data: pd.DataFrame, fit: Dict) -> pd.DataFrame:
    """Instrument gradient of a fit at other exposures, e.g. twilight I at science frames

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of the same channel, from :func:`prepare_gradients`.
    fit : dict
        Fit from :func:`fit_gradient_model`. Exposures out of the nights of a
        segment take the instrument gradient of the segment they fall in.

    Returns
    -------
    pandas.DataFrame
        `data` with inst_x and inst_y (%), NaN out of the segments of the fit.
    """
    data = data.copy()
    data["inst_x"], data["inst_y"] = np.nan, np.nan
    instrument = fit["instrument"]
    if instrument.empty:
        data["inst_x"], data["inst_y"] = 0.0, 0.0
        return data
    segment = _segments(data["mjd"].to_numpy(float), fit["breaks"]) if not isinstance(fit["breaks"], str) else None
    if segment is None:
        raise ValueError("cannot predict an instrument gradient fitted per night at other exposures")
    rotation = INSTRUMENT_FRAMES[fit["frame"]][1]
    phi = np.radians(rotation * data["phi"].to_numpy(float)) if rotation else np.zeros(len(data))
    for row in instrument.itertuples():
        m = segment == row.segment
        data.loc[m, "inst_x"] = np.cos(phi[m]) * row.gx - np.sin(phi[m]) * row.gy
        data.loc[m, "inst_y"] = np.sin(phi[m]) * row.gx + np.cos(phi[m]) * row.gy
    return data


def cross_check(data: pd.DataFrame, fit: Dict, sky: Dict[str, str]) -> Dict:
    """How well the instrument gradient of one source explains the gradients of another

    Removes the predicted instrument gradient from the gradients of `data`
    and fits only their sky gradient, without and with the correction.

    Parameters
    ----------
    data : pandas.DataFrame
        Gradients of one channel and source, from :func:`prepare_gradients`.
    fit : dict
        Fit of the instrument gradient on another source, from :func:`fit_gradient_model`.
    sky : dict
        Sky gradients of `data`, {direction column: amplitude}.

    Returns
    -------
    dict
        n, rms of the gradients around zero (raw) and after the correction
        (corrected), rms of the sky-only fit without (sky_raw) and with the
        correction (sky_corrected), and the alignment of the measured gradients
        (g) and of the gradients minus the twilight instrument gradient (g − I),
        see :func:`residual_alignment`.
    """
    source = data["source"].iloc[0]
    predicted = predict_instrument(data, fit)
    predicted = predicted[np.isfinite(predicted["inst_x"])].reset_index(drop=True)
    if predicted.empty:
        return {"n": 0}
    corrected = predicted.copy()
    corrected["gx"], corrected["gy"] = predicted["gx"] - predicted["inst_x"], predicted["gy"] - predicted["inst_y"]
    raw_fit = fit_gradient_model(predicted, frame="none", sky={source: sky}, crossval=False)
    corrected_fit = fit_gradient_model(corrected, frame="none", sky={source: sky}, crossval=False)
    return {"n": len(predicted),
            "raw": float(np.sqrt(np.mean(predicted["gx"] ** 2 + predicted["gy"] ** 2))),
            "corrected": float(np.sqrt(np.mean(corrected["gx"] ** 2 + corrected["gy"] ** 2))),
            "sky_raw": raw_fit["rms"], "sky_corrected": corrected_fit["rms"],
            "alignment": residual_alignment({"data": predicted})}


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def _num(value, fmt):
    return "-" if value is None or not np.isfinite(value) else format(value, fmt)


def figure_vectors(fit: Dict, title: str = None) -> go.Figure:
    """Measured gradients in IFU coordinates by segment, with the instrument gradient of each segment"""
    data, instrument = fit["data"], fit["instrument"]
    segment = _segments(data["mjd"].to_numpy(float), fit["breaks"]) if not isinstance(fit["breaks"], str) else np.zeros(len(data), int)
    fig = go.Figure()
    segments = np.unique(segment)
    colors = SERIES if len(segments) <= len(SERIES) else sample_colorscale(SEQUENTIAL, np.linspace(0.25, 1, len(segments)))
    for i, k in enumerate(segments):
        sel = data[segment == k]
        color = colors[i]
        name = f"{int(sel['mjd'].min())}-{int(sel['mjd'].max())}"
        fig.add_trace(go.Scatter(x=sel["gx"], y=sel["gy"], mode="markers", name=name, legendgroup=name, showlegend=False,
                                 marker=dict(size=6, color=color, opacity=0.45),
                                 customdata=np.stack([sel["mjd"], sel["expnum"]], axis=-1),
                                 hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>(%{x:.2f}, %{y:.2f})%<extra></extra>"))
        row = instrument[instrument["segment"] == k] if not instrument.empty else instrument
        if len(row):
            row = row.iloc[0]
            fig.add_trace(go.Scatter(x=[0, row.gx], y=[0, row.gy], mode="lines+markers", name=name, legendgroup=name,
                                     line=dict(color=color, width=3), marker=dict(size=[0, 14], symbol="diamond", color=color,
                                                                                    line=dict(width=2, color=THEME["ink"])),
                                     error_x=dict(type="data", array=[0, row.gx_err], visible=True, color=THEME["ink2"]),
                                     error_y=dict(type="data", array=[0, row.gy_err], visible=True, color=THEME["ink2"]),
                                     hovertemplate=f"<b>I {name}</b><br>{row.strength:.2f}% @ {row.direction:.0f}°<extra></extra>"))
    limit = float(np.nanpercentile(np.abs(np.r_[data["gx"], data["gy"]]), 98)) * 1.15 if len(data) else 1.0
    fig.update_layout(base_layout(440, xaxis_title="g<sub>x</sub> (%)", yaxis_title="g<sub>y</sub> (%)", title=title,
                                  legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top", font=dict(size=11, color=THEME["ink2"]),
                                              bgcolor="rgba(0,0,0,0)", title=dict(text="I per segment<br>(dots: g)"))))
    style_axes(fig)
    fig.update_xaxes(range=[-limit, limit], zeroline=True, constrain="domain")
    fig.update_yaxes(range=[-limit, limit], zeroline=True, scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def figure_nightly_instrument(fits: Dict[str, Dict], breaks: Sequence[int] = ()) -> go.Figure:
    """Instrument gradient fitted per night (sky model shared by all nights), strength and direction against the night

    The error bars are the jackknife errors over nights when a segment has at
    least 3 nights, else the formal errors, which ignore that the exposures of
    a night share their errors. For each exposure, spread within its night:
    the measured gradient (g, gray dots), the measured gradient minus the
    instrument gradient of its night (g - I, open circles) and, in the
    direction panel, the direction of its sky gradient (crosses): the Sun for
    twilights, the horizon for dome flats and science frames.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for i, (label, fit) in enumerate(fits.items()):
        instrument = fit["instrument"]
        if instrument.empty:
            continue
        color = SERIES[i % len(SERIES)]
        # jackknife errors where the segment has enough nights, else the formal ones (always for one segment per night)
        sx = np.where(np.isfinite(instrument["gx_err"]), instrument["gx_err"], instrument["gx_sig"])
        sy = np.where(np.isfinite(instrument["gy_err"]), instrument["gy_err"], instrument["gy_sig"])
        err, direction_err = np.array([_polar_errors(gx, gy, ex, ey) for gx, gy, ex, ey in zip(instrument["gx"], instrument["gy"], sx, sy)]).T
        common = dict(mode="markers", legendgroup=label, marker=dict(size=7, color=color),
                      customdata=np.stack([instrument["mjd_first"], instrument["n"]], axis=-1))
        fig.add_trace(go.Scatter(x=instrument["mjd_first"], y=instrument["strength"], name=label,
                                 error_y=dict(type="data", array=err, visible=True, thickness=1),
                                 hovertemplate="MJD %{customdata[0]:.0f} · %{customdata[1]} exposures<br>%{y:.2f}%<extra></extra>", **common), row=1, col=1)
        fig.add_trace(go.Scatter(x=instrument["mjd_first"], y=instrument["direction"], name=label, showlegend=False,
                                 error_y=dict(type="data", array=np.minimum(direction_err, 180), visible=True, thickness=1),
                                 hovertemplate="MJD %{customdata[0]:.0f} · %{customdata[1]} exposures<br>%{y:.0f}°<extra></extra>", **common), row=2, col=1)
        # g and g - I of each exposure, with the I of its night, spread within the night so they don't overlap
        data = fit["data"]
        if "inst_x" in data.columns and len(data):
            rank = data.groupby("mjd").cumcount().to_numpy()
            size = data.groupby("mjd")["mjd"].transform("size").to_numpy()
            x = data["mjd"].to_numpy(float) + 0.6 * ((rank + 1) / (size + 1) - 0.5)
            rx, ry = data["gx"] - data["inst_x"], data["gy"] - data["inst_y"]
            custom = np.stack([data["mjd"].astype(int), data["expnum"].astype(int)], axis=-1)
            measured = dict(mode="markers", legendgroup="g", customdata=custom, marker=dict(size=4, color=THEME["muted"], opacity=0.7))
            fig.add_trace(go.Scatter(x=x, y=np.hypot(data["gx"], data["gy"]), name="g per exposure",
                                     hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>g %{y:.2f}%<extra></extra>",
                                     **measured), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=np.degrees(np.arctan2(data["gy"], data["gx"])), name="g", showlegend=False,
                                     hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>g towards %{y:.0f}°<extra></extra>",
                                     **measured), row=2, col=1)
            exposures = dict(mode="markers", legendgroup="g − I", customdata=custom,
                             marker=dict(size=5, color=SERIES[1], opacity=0.55, symbol="circle-open", line=dict(width=1.2)))
            fig.add_trace(go.Scatter(x=x, y=np.hypot(rx, ry), name="g − I per exposure",
                                     hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>g − I %{y:.2f}%<extra></extra>",
                                     **exposures), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=np.degrees(np.arctan2(ry, rx)), name="g − I", showlegend=False,
                                     hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>g − I towards %{y:.0f}°<extra></extra>",
                                     **exposures), row=2, col=1)
            reference = np.array([data[SOURCE_SKY_REFERENCE[s]].iloc[j] if SOURCE_SKY_REFERENCE.get(s) in data.columns else np.nan
                                  for j, s in enumerate(data["source"])], dtype=float)
            nouns = sorted({"Sun" if SOURCE_SKY_REFERENCE.get(s) == "dir_sun" else "horizon" for s in set(data["source"])})
            if np.isfinite(reference).any():
                fig.add_trace(go.Scatter(x=x, y=reference, mode="markers", name=f"towards the {' or '.join(nouns)}", customdata=custom,
                                         marker=dict(size=7, symbol="x-thin", color=THEME["ink"], line=dict(width=1.5, color=THEME["ink"])),
                                         hovertemplate="MJD %{customdata[0]} · exposure %{customdata[1]}<br>sky towards %{y:.0f}°<extra></extra>"),
                              row=2, col=1)
    nights = np.concatenate([f["data"]["mjd"].to_numpy(float) for f in fits.values() if len(f["data"])] or [np.array([0.0])])
    pad = max(2.0, 0.03 * np.ptp(nights))
    lo, hi = nights.min() - pad, nights.max() + pad
    for mjd in breaks:
        if lo <= mjd <= hi:
            fig.add_vline(x=mjd, line=dict(color=THEME["muted"], width=1, dash="dot"))
    fig.update_xaxes(range=[lo, hi])
    fig.update_layout(base_layout(560))
    style_axes(fig)
    fig.update_yaxes(title_text="Strength (%)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Direction (°)", range=[-180, 180], dtick=90, row=2, col=1)
    fig.update_xaxes(title_text="MJD", row=2, col=1)
    return fig


def _introduction(sources: Sequence[str]) -> str:
    """Introduction of the dashboard, naming only the sources of exposures of the run"""
    nouns = {"twilight": "twilights", "dome": "dome flats", "science": "science frames"}
    present = [s for s in SOURCE_SKY_REFERENCE if s in sources]
    sky = (f"the direction of {'the Sun' if SOURCE_SKY_REFERENCE[present[0]] == 'dir_sun' else 'the horizon'}" if len(present) == 1 else
           " and ".join(f"the direction of {'the Sun' if SOURCE_SKY_REFERENCE[s] == 'dir_sun' else 'the horizon'} for {nouns[s]}" for s in present))
    flats = [nouns[s] for s in present if s != "science"]
    if flats and "science" in present:
        kmirror = (f"In {' and '.join(flats)} the K-mirror does not move, so only the science frames tell an IFU-fixed gradient from one "
                   f"fixed before the K-mirror (flagged as degenerate otherwise).")
    elif flats:
        kmirror = (f"In {' and '.join(flats)} the K-mirror does not move, so they cannot tell an IFU-fixed gradient from one fixed "
                   f"before the K-mirror: those models are flagged as degenerate, and science frames are needed to tell them apart.")
    else:
        kmirror = "In science frames the K-mirror follows the field rotation, which tells an IFU-fixed gradient from one fixed before it."
    joint = (f" The {' and the '.join(nouns[s] for s in present)} are fitted separately and together (joint), sharing the instrument "
             f"gradient." if len(present) > 1 else "")
    return ("<p>Tests whether the gradient that the pipeline fits across the science IFU holds an instrument gradient, fixed in "
            "the IFU or before the K-mirror, besides the sky gradient, fixed on the sky. Each model is fitted to the gradient vectors "
            "of all the exposures of a channel; the instrument gradient is shared by all the exposures of a segment between breaks, "
            f"the sky gradient follows {sky} across the IFU.{joint} Lower BIC is better; the leave-one-night-out rms checks that the "
            f"models predict nights they did not see. {kmirror} Every quantity is defined under "
            '<a href="#definitions">How the quantities are computed</a>.</p>')


def _sky_nouns(data: pd.DataFrame) -> str:
    """The sky direction of each source in `data`, e.g. 'towards the Sun for twilights, towards the horizon for science frames'"""
    nouns = {"twilight": "twilights", "dome": "dome flats", "science": "science frames"}
    present = [s for s in SOURCE_SKY_REFERENCE if s in set(data["source"])]
    parts = [f"towards {'the Sun' if SOURCE_SKY_REFERENCE[s] == 'dir_sun' else 'the horizon'}" + (f" for {nouns[s]}" if len(present) > 1 else "")
             for s in present]
    return ", ".join(parts)


def _nightly_or_best(nightly: Dict, best: Dict) -> Tuple[Dict[str, Dict], str]:
    """The fit with one I per night, or the best fit drawn night by night when the sky shared by the nights is not constrained

    With one I per night, the sky gradient is only constrained by how it
    changes between nights; with few nights (or nights with the same
    geometry) its shared amplitudes are not determined and I per night takes
    any value. Then the I of the segments of the best model is shown at each
    night instead, with the g - I of the best model.

    Returns
    -------
    tuple
        The fits to draw, {label: fit}, for :func:`figure_nightly_instrument`,
        and a note for the page, empty when the fit per night is used.
    """
    shared = nightly["sky"][nightly["sky"]["group"] == "all"] if not nightly["sky"].empty else nightly["sky"]
    worst = float(shared["sigma"].max()) if len(shared) else 0.0
    if not nightly["instrument"].empty and np.isfinite(worst) and worst <= nightly["rms"]:
        return {"I per night": nightly}, ""
    data = best["data"]
    segment = _segments(data["mjd"].to_numpy(float), best["breaks"])
    per_night = (pd.DataFrame({"mjd": data["mjd"].to_numpy(int), "segment": segment}).groupby("mjd")
                 .agg(segment=("segment", "first"), n=("segment", "size")).reset_index())
    per_night = per_night.merge(best["instrument"].drop(columns=["n", "nights"]), on="segment", how="inner")
    per_night["mjd_first"] = per_night["mjd"].astype(float)
    note = (f" With {nightly['nights']} nights the sky gradient shared by the nights is not constrained (formal error of its "
            f"amplitude {worst:.1f}%, against an rms of {nightly['rms']:.2f}%), so one I per night cannot be fitted: the plot shows "
            f"the I of the best model at each night, and g − I with it.")
    return {"I of the best model": {**best, "instrument": per_night}}, note


def figure_direction_scan(scan: pd.DataFrame) -> go.Figure:
    """rms of the fits against the offset added to the sky direction, with and without an instrument gradient"""
    fig = go.Figure()
    for mirrored, dash in ((False, "solid"), (True, "dot")):
        sel = scan[scan["mirrored"] == mirrored]
        suffix = " (mirrored)" if mirrored else ""
        fig.add_trace(go.Scatter(x=sel["offset"], y=sel["rms_inst"], mode="lines", name=f"sky + I{suffix}", line=dict(color=SERIES[0], dash=dash)))
        fig.add_trace(go.Scatter(x=sel["offset"], y=sel["rms_sky"], mode="lines", name=f"sky only{suffix}", line=dict(color=SERIES[1], dash=dash)))
    fig.add_vline(x=0, line=dict(color=THEME["muted"], width=1, dash="dot"))
    fig.update_layout(base_layout(420, xaxis_title="Offset added to the direction of the Sun (°)", yaxis_title="rms (%)"))
    style_axes(fig)
    fig.update_xaxes(dtick=45)
    fig.update_yaxes(rangemode="tozero")
    return fig


def figure_residuals(fit: Dict) -> go.Figure:
    """Residual of the best fit against the night"""
    data = fit["data"]
    fig = go.Figure(go.Scatter(x=data["mjd"], y=np.hypot(data["res_x"], data["res_y"]), mode="markers",
                               marker=dict(size=6, color=SERIES[0], opacity=0.6),
                               customdata=data["expnum"], hovertemplate="exposure %{customdata}<br>%{y:.2f}%<extra></extra>"))
    fig.update_layout(base_layout(320, xaxis_title="MJD", yaxis_title="|r| = |g − I − sky| (%)"))
    style_axes(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


FRAME_SHORT = {"none": "none", "ifu": "IFU", "kmirror": "K-mirror +φ", "kmirror-": "K-mirror −φ"}


# sky directions drawn in the dashboard: column, label and color
SKY_REFERENCES = (("dir_sun", "Sun", SERIES[1]), ("dir_horizon", "Horizon", SERIES[2]), ("dir_moon", "Moon", THEME["muted"]))

# sky directions as written in the report: 'towards the Sun', 'from the horizon', ...
SKY_NOUN = {"Sun": "the Sun", "Horizon": "the horizon", "Moon": "the Moon"}

# sky direction the gradient left in g - I should follow for each source
SOURCE_SKY_REFERENCE = {"twilight": "dir_sun", "dome": "dir_horizon", "science": "dir_horizon"}


def ifu_fiber_positions() -> Tuple[np.ndarray, np.ndarray]:
    """Positions (mm) of the science IFU fibers, centered and scaled to the IFU edge (radius 1)

    From the fibermap in lvmcore; without it, a hexagonal grid of the same
    size, as only the shape of the IFU matters for the maps.
    """
    try:
        from lvmdrp.main import read_fibermap

        fibermap = read_fibermap(as_table=True)
        sci = np.array([str(t) == "Sci" for t in fibermap["telescope"]])
        x, y = np.asarray(fibermap["xpmm"], dtype=float)[sci], np.asarray(fibermap["ypmm"], dtype=float)[sci]
    except Exception as e:
        log.warning(f"cannot read the fibermap ({type(e).__name__}: {e}), drawing the IFU as a hexagonal grid")
        i, j = np.meshgrid(np.arange(-25, 26), np.arange(-25, 26))
        x, y = (i + 0.5 * j).ravel(), (np.sqrt(3) / 2 * j).ravel()
        keep = (np.abs(j.ravel()) <= 24) & (np.abs(x) + np.abs(y) / np.sqrt(3) <= 24.5) & (np.abs(y) <= 24 * np.sqrt(3) / 2)
        x, y = x[keep], y[keep]
    x, y = x - x.mean(), y - y.mean()
    radius = np.max(np.hypot(x, y))
    return x / radius, y / radius


def _mjd_list(value) -> List[int]:
    """MJDs of a flavor of the calibration epochs file: an integer, a comma separated list or a range 'MJD1-MJD2'"""
    if value is None:
        return []
    mjds = []
    for item in str(value).split(","):
        item = item.strip()
        if "-" in item:
            first, last = (int(v) for v in item.split("-"))
            mjds.extend(range(first, last + 1))
        elif item:
            mjds.append(int(item))
    return mjds


def calibration_epoch_of(nights: Sequence[int], epochs: Dict[int, Dict], flavor: str = "twilight") -> Dict[int, int]:
    """Calibration epoch of each night: the one whose `flavor` MJDs include it, else the last epoch starting on or before it"""
    starts = sorted(epochs)
    listed = {night: mjd for mjd in starts for night in _mjd_list((epochs[mjd] or {}).get("flavors", {}).get(flavor))}
    result = {}
    for night in sorted(set(int(n) for n in nights)):
        if night in listed:
            result[night] = listed[night]
        else:
            before = [mjd for mjd in starts if mjd <= night]
            result[night] = before[-1] if before else starts[0]
    return result


BROWSER_SOURCES = {"twilight": "Twilights", "dome": "Dome flats"}

# time zone of LCO, for the local time of the exposures (Chilean time, with daylight saving)
LCO_TIMEZONE = "America/Santiago"

BROWSER_STYLE = """<style>
.eb-controls { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; font-size: 13px; color: var(--ink-2); }
.eb-controls label { display: inline-flex; gap: 8px; align-items: center; }
.eb-controls select { font: 13px var(--sans); color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px; padding: 4px 8px; }
.eb-legend { display: flex; flex-wrap: wrap; gap: 8px 28px; align-items: center; font-size: 12px; color: var(--ink-2); }
.eb-bar { display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); }
.eb-bar span.ramp { display: inline-block; width: 140px; height: 10px; border-radius: 3px; border: 1px solid var(--border); }
.eb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 12px; }
.eb-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px; display: grid; gap: 6px; min-width: 0; }
.eb-title { font-size: 12px; color: var(--ink); font-family: var(--mono); }
.eb-panels { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
.eb-panel { display: grid; gap: 2px; justify-items: center; min-width: 0; }
.eb-panel canvas { width: 100%; max-width: 120px; aspect-ratio: 1; }
.eb-caption { font-size: 11px; color: var(--ink-2); text-align: center; line-height: 1.3; }
.eb-card.departs { border: 2px solid var(--warn); }
.eb-card.departs .eb-flag { color: var(--warn); font-weight: 600; }
.eb-flags { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; font-size: 12px; color: var(--ink-2); }
.eb-flags button { font: 12px var(--mono); color: var(--warn); background: var(--surface); border: 1px solid var(--warn); border-radius: 999px; padding: 2px 10px; cursor: pointer; }
.eb-flags button.current { color: var(--surface); background: var(--warn); }
</style>"""

BROWSER_SCRIPT = """<script>
(function () {
  const DATA = @@DATA@@;
  const $ = (id) => document.getElementById(id);
  const source = $("eb-source"), channel = $("eb-channel"), epoch = $("eb-epoch");
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const DEG = "\\u00b0", MINUS = "\\u2212";
  function isDark() {
    const theme = document.documentElement.getAttribute("data-theme");
    return theme ? theme === "dark" : media.matches;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function hex(c) { return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); }
  function colormap(t) {
    const stops = isDark() ? DATA.div_dark : DATA.div;
    t = Math.min(1, Math.max(0, t));
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const f = (t - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
        const a = hex(stops[i - 1][1]), b = hex(stops[i][1]);
        return "rgb(" + a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",") + ")";
      }
    }
    return stops[stops.length - 1][1];
  }
  function ramp() {
    const n = 11, colors = [];
    for (let i = 0; i < n; i++) colors.push(colormap(i / (n - 1)));
    return "linear-gradient(90deg," + colors.join(",") + ")";
  }
  function polar(vx, vy) { return [Math.hypot(vx, vy), Math.atan2(vy, vx) * 180 / Math.PI]; }
  function wrap(a) { return ((a + 180) % 360 + 360) % 360 - 180; }
  function fmt(vx, vy) { const [s, d] = polar(vx, vy); return s.toFixed(2) + "% @ " + d.toFixed(0) + DEG; }

  function draw(canvas, vx, vy, limit, arrows, divide) {
    const ratio = window.devicePixelRatio || 1, size = 120;
    canvas.width = size * ratio; canvas.height = size * ratio;
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const scale = size / 2 / 1.25, cx = size / 2, cy = size / 2, radius = Math.max(0.8, 0.5 * DATA.spacing * scale);
    for (let i = 0; i < DATA.x.length; i++) {
      // the gradients multiply: the g - I map is the measured illumination divided by the instrument model
      const v = divide ? 100 * ((1 + (divide[0] * DATA.x[i] + divide[1] * DATA.y[i]) / 100) /
                                (1 + (divide[2] * DATA.x[i] + divide[3] * DATA.y[i]) / 100) - 1)
                       : vx * DATA.x[i] + vy * DATA.y[i];
      ctx.fillStyle = colormap(0.5 + 0.5 * v / limit);
      ctx.beginPath();
      ctx.arc(cx + DATA.x[i] * scale, cy - DATA.y[i] * scale, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
    for (const [direction, color, dashed] of arrows) {
      if (direction === null || !isFinite(direction)) continue;
      const a = direction * Math.PI / 180, ex = cx + 1.15 * scale * Math.cos(a), ey = cy - 1.15 * scale * Math.sin(a);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.6;
      ctx.setLineDash(dashed ? [3, 3] : []);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
      const h = 6, b = Math.atan2(cy - ey, ex - cx);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - h * Math.cos(b - 0.45), ey + h * Math.sin(b - 0.45));
      ctx.lineTo(ex - h * Math.cos(b + 0.45), ey + h * Math.sin(b + 0.45));
      ctx.closePath(); ctx.fill();
    }
  }

  function key() { return source.value + "|" + channel.value; }
  // angle from the sky reference to g - I, and whether the exposure departs from it
  function departure(set, r) {
    const rx = r[5] - r[7], ry = r[6] - r[8], reference = set.reference === "sun" ? r[9] : r[10];
    if (reference === null) return [null, false];
    const offset = Math.abs(wrap(polar(rx, ry)[1] - reference));
    return [offset, offset > DATA.departure.angle && Math.hypot(rx, ry) >= DATA.departure.strength];
  }
  function epochStats(set) {
    const stats = {};
    for (const r of set.rows) {
      const s = stats[r[2]] || (stats[r[2]] = { n: 0, departs: 0 });
      s.n += 1; if (departure(set, r)[1]) s.departs += 1;
    }
    for (const s of Object.values(stats)) {
      s.flagged = s.departs >= DATA.departure.min_exposures && s.departs >= DATA.departure.fraction * s.n;
    }
    return stats;
  }
  function fillChannels() {
    const current = channel.value;
    channel.innerHTML = "";
    for (const k of Object.keys(DATA.sets)) {
      const [s, c] = k.split("|");
      if (s === source.value) channel.add(new Option(c, c));
    }
    if ([...channel.options].some((o) => o.value === current)) channel.value = current;
  }
  function fillEpochs() {
    const current = epoch.value, set = DATA.sets[key()], stats = epochStats(set);
    epoch.innerHTML = "";
    for (const e of Object.keys(stats).sort()) {
      const s = stats[e];
      epoch.add(new Option((s.flagged ? "\\u26a0 " : "") + (DATA.epochs[e] || e) + " (" + s.n + (s.departs ? ", " + s.departs + " depart" : "") + ")", e));
    }
    if ([...epoch.options].some((o) => o.value === current)) epoch.value = current;
    // the epochs where g - I departs from its sky direction, as buttons
    const flagged = Object.keys(stats).sort().filter((e) => stats[e].flagged), box = $("eb-flags");
    box.innerHTML = flagged.length ? "<span>g " + MINUS + " I departs from " + (set.reference === "sun" ? "the Sun" : "the horizon") + " in:</span>"
                                   : "<span>No epoch where g " + MINUS + " I departs from " + (set.reference === "sun" ? "the Sun" : "the horizon") + ".</span>";
    for (const e of flagged) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = e + " \\u00b7 " + stats[e].departs + "/" + stats[e].n;
      button.title = DATA.epochs[e] || e;
      button.addEventListener("click", () => { epoch.value = e; render(); });
      box.appendChild(button);
    }
  }
  function render() {
    const set = DATA.sets[key()], rows = set.rows.filter((r) => String(r[2]) === epoch.value);
    let limit = 1e-3, limitRes = 1e-3;
    for (const r of rows) {
      limit = Math.max(limit, Math.hypot(r[5], r[6]), Math.hypot(r[7], r[8]));
      limitRes = Math.max(limitRes, Math.hypot(r[5] - r[7], r[6] - r[8]));
    }
    const gradient = ramp();
    $("eb-legend").innerHTML =
      '<span class="eb-bar">g, I <span>' + MINUS + limit.toFixed(1) + '%</span><span class="ramp" style="background:' + gradient +
      '"></span><span>+' + limit.toFixed(1) + '%</span></span>' +
      '<span class="eb-bar">g ' + MINUS + ' I <span>' + MINUS + limitRes.toFixed(1) + '%</span><span class="ramp" style="background:' + gradient +
      '"></span><span>+' + limitRes.toFixed(1) + '%</span></span>' +
      '<span>arrows: <b style="color:' + css("--warn") + '">Sun</b>, <b style="color:' + css("--ok") + '">horizon</b>, <b style="color:' +
      css("--muted") + '">Moon</b> (when up)</span><span>' + set.model + '</span>';
    const grid = $("eb-grid");
    grid.innerHTML = "";
    const ink = css("--ink");
    for (const r of rows) {
      const [expnum, mjd, , exptime, sunAlt, gx, gy, ix, iy, dSun, dHorizon, dMoon, moonAlt, local, illum] = r;
      const rx = gx - ix, ry = gy - iy, reference = set.reference === "sun" ? dSun : dHorizon;
      const [offset, departs] = departure(set, r);
      const card = document.createElement("div");
      card.className = departs ? "eb-card departs" : "eb-card";
      const moonUp = moonAlt !== null && moonAlt > 0;
      const moon = moonAlt === null ? "" : moonUp ? "Moon up " + moonAlt.toFixed(0) + DEG + (illum === null ? "" : ", " + (100 * illum).toFixed(0) + "% lit")
                                                  : "Moon down";
      card.innerHTML = '<div class="eb-title">' + expnum + " \\u00b7 MJD " + mjd + (local === null ? "" : " \\u00b7 " + local + " local") +
        (exptime === null ? "" : " \\u00b7 " + exptime + " s") + "<br>" + (sunAlt === null ? "" : "Sun " + sunAlt.toFixed(1) + DEG) +
        (moon ? (sunAlt === null ? "" : " \\u00b7 ") + moon : "") + '</div><div class="eb-panels"></div>';
      // angle from each sky direction to g - I, the closest one highlighted in its color
      const near = [["Sun", dSun, css("--warn")], ["horizon", dHorizon, css("--ok")], ["Moon", moonUp ? dMoon : null, css("--muted")]]
        .filter((d) => d[1] !== null).map(([name, d, color]) => [name, Math.abs(wrap(polar(rx, ry)[1] - d)), color]);
      const closest = near.length ? near.reduce((a, b) => (b[1] < a[1] ? b : a))[0] : null;
      const offsets = near.map(([name, o, color]) => name === closest ? '<b style="color:' + color + '">' + name + " " + o.toFixed(0) + DEG + "</b>"
                                                                      : name + " " + o.toFixed(0) + DEG).join(" \\u00b7 ");
      const panels = card.querySelector(".eb-panels");
      const specs = [
        ["g " + fmt(gx, gy), gx, gy, limit, [[polar(gx, gy)[1], ink, false]]],
        ["I " + fmt(ix, iy), ix, iy, limit, [[polar(ix, iy)[1], ink, false]]],
        ["g " + MINUS + " I " + fmt(rx, ry) + (offsets ? '<br><span class="eb-flag">from ' + offsets + "</span>" : ""),
         rx, ry, limitRes, [[dSun, css("--warn"), true], [dHorizon, css("--ok"), true], [moonAlt !== null && moonAlt > 0 ? dMoon : null, css("--muted"), true]],
         [gx, gy, ix, iy]],
      ];
      for (const [caption, vx, vy, lim, arrows, divide] of specs) {
        const panel = document.createElement("div");
        panel.className = "eb-panel";
        const canvas = document.createElement("canvas");
        panel.appendChild(canvas);
        const label = document.createElement("div");
        label.className = "eb-caption"; label.innerHTML = caption;
        panel.appendChild(label);
        panels.appendChild(panel);
        draw(canvas, vx, vy, lim, arrows, divide);
      }
      grid.appendChild(card);
    }
    for (const button of $("eb-flags").querySelectorAll("button")) {
      button.classList.toggle("current", button.textContent.split(" ")[0] === epoch.value);
    }
  }
  for (const s of [...new Set(Object.keys(DATA.sets).map((k) => k.split("|")[0]))]) source.add(new Option(DATA.sources[s] || s, s));
  source.addEventListener("change", () => { fillChannels(); fillEpochs(); render(); });
  channel.addEventListener("change", () => { fillEpochs(); render(); });
  epoch.addEventListener("change", render);
  fillChannels(); fillEpochs(); render();
  media.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>"""


def _local_times(data: pd.DataFrame, timezone: str = LCO_TIMEZONE) -> Dict[int, str]:
    """Local time at LCO (HH:MM) of the middle of each exposure, from its UTC ``time``, by exposure number"""
    if "time" not in data.columns:
        return {}
    times = pd.to_datetime(data["time"], utc=True, errors="coerce")
    try:
        times = times.dt.tz_convert(timezone)
    except Exception as e:
        log.warning(f"cannot convert the times to {timezone} ({type(e).__name__}: {e}), using UTC-4")
        times = times - pd.Timedelta(hours=4)
    return {int(e): t.strftime("%H:%M") for e, t in zip(data["expnum"], times) if not pd.isna(t)}


def epoch_browser(fits: Dict[Tuple[str, str], Dict], positions: Tuple[np.ndarray, np.ndarray], epochs: Dict[int, Dict] = None,
                  departure_angle: float = 45.0, departure_strength: float = 0.5, departure_fraction: float = 0.25,
                  departure_min: int = 2) -> str:
    """HTML of the maps across the IFU of every twilight (and dome flat) exposure, grouped by calibration epoch

    The maps are drawn in the browser from the gradient vectors of each
    exposure (measured, instrument model and their difference) and the fiber
    positions, so that all the exposures fit in the page; the source, channel
    and calibration epoch are chosen with dropdowns.

    Parameters
    ----------
    fits : dict
        Best fit of each (source, channel), from :func:`fit_gradient_model`;
        only the 'twilight' and 'dome' sources are shown.
    positions : tuple[np.ndarray, np.ndarray]
        Fiber positions scaled to the IFU edge, from :func:`ifu_fiber_positions`.
    epochs : dict, optional
        Calibration epochs (see :func:`lvmdrp.qa.fiberflats.load_calibration_epochs`).
        Without them, the exposures are grouped by the segments of the fit.
    departure_angle, departure_strength : float, optional
        An exposure departs from its sky direction (the Sun for twilights, the
        horizon for dome flats) when g - I points more than `departure_angle`
        degrees away from it and is at least `departure_strength` % strong,
        so that its direction is measured. Defaults are 45 deg and 0.5%.
    departure_fraction, departure_min : float and int, optional
        An epoch is flagged when at least this fraction of its exposures, and
        at least `departure_min` of them, depart. Defaults are 0.25 and 2.

    Returns
    -------
    str
        The HTML section, empty if there is nothing to show.
    """
    def value(row, column):
        v = getattr(row, column, None)
        return None if v is None or not np.isfinite(float(v)) else round(float(v), 3)

    sets, labels = {}, {}
    for (source, channel), fit in fits.items():
        if source not in BROWSER_SOURCES:
            continue
        data = fit["data"][fit["data"]["source"] == source]
        if data.empty:
            continue
        nights = data["mjd"].astype(int).to_numpy()
        if epochs:
            epoch_of = calibration_epoch_of(nights, epochs, flavor=source)
            for mjd in set(epoch_of.values()):
                epoch = epochs[mjd] or {}
                listed = _mjd_list(epoch.get("flavors", {}).get(source))
                labels[str(mjd)] = f"{mjd} · {epoch.get('trigger', '')}" + (f" · {source}s {', '.join(map(str, listed))}" if listed else "")
            groups = [str(epoch_of[n]) for n in nights]
        else:
            edges = _segment_edges(nights.astype(float), fit["breaks"])
            segment = _segments(nights.astype(float), fit["breaks"])
            groups = [f"{int(edges[k][0])}" for k in segment]
            labels.update({g: f"segment from {g}" for g in groups})
        local = _local_times(data)
        rows = []
        order = np.lexsort((data["expnum"].to_numpy(), nights))
        for i in order:
            row = next(data.iloc[[i]].itertuples())
            illum = value(row, "moon_illum")
            rows.append([int(row.expnum), int(row.mjd), groups[i], value(row, "exptime"), value(row, "sun_alt"),
                         round(row.gx, 4), round(row.gy, 4), round(row.inst_x, 4), round(row.inst_y, 4),
                         value(row, "dir_sun"), value(row, "dir_horizon"), value(row, "dir_moon"), value(row, "moon_alt"),
                         local.get(int(row.expnum)),
                         # illuminated fraction of the Moon; some tables give it in percent
                         None if illum is None else round(illum / 100 if illum > 1 else illum, 3)])
        last = fit["instrument"].iloc[-1] if not fit["instrument"].empty else None
        sets[f"{source}|{channel}"] = {
            "rows": rows, "reference": "sun" if SOURCE_SKY_REFERENCE[source] == "dir_sun" else "horizon",
            "model": f"I {INSTRUMENT_FRAMES[fit['frame']][0]}, breaks {_label_breaks(fit['breaks'])}"
                     + (f", last segment {last.strength:.2f}% @ {last.direction:.0f}°" if last is not None else "")}
    if not sets:
        return ""
    x, y = positions
    tree = cKDTree(np.column_stack([x, y]))
    spacing = float(np.median(tree.query(np.column_stack([x, y]), k=2)[0][:, 1]))
    payload = {"x": np.round(x, 4).tolist(), "y": np.round(y, 4).tolist(), "spacing": spacing, "sets": sets, "epochs": labels,
               "sources": BROWSER_SOURCES, "div": DIVERGING, "div_dark": DIVERGING_DARK,
               "departure": {"angle": departure_angle, "strength": departure_strength, "fraction": departure_fraction,
                             "min_exposures": departure_min}}
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    nouns = " and ".join(BROWSER_SOURCES[s].lower() for s in BROWSER_SOURCES if any(k.startswith(s + "|") for k in sets))
    references = ", ".join(f"the {'Sun' if SOURCE_SKY_REFERENCE[s] == 'dir_sun' else 'horizon'}"
                           + (f" for {BROWSER_SOURCES[s].lower()}" if len({k.split('|')[0] for k in sets}) > 1 else "")
                           for s in BROWSER_SOURCES if any(k.startswith(s + "|") for k in sets))
    return (BROWSER_STYLE
            + f'<section id="epoch-browser"><h2>{nouns.capitalize()} by calibration epoch</h2>'
            f"<p>Every exposure of the chosen calibration epoch across the science IFU: the plane measured by the pipeline (g), the "
            f"instrument model of the best fit at that exposure (I) and the measured illumination divided by it (g − I), with the arrows of the Sun, the "
            f"horizon and the Moon in the last panel. g and I share a color scale and g − I has its own, both set by the exposures "
            f"of the epoch shown. Exposures are assigned to the epoch whose {nouns} MJDs include their night, or else to the last "
            f"epoch before it.</p><p>An exposure departs from its sky direction ({references}) "
            f"when g − I points more than {departure_angle:.0f}° away from it and is at least {departure_strength:g}% strong, so "
            f"that its direction is measured; its card has an orange border. Each card gives the local time at LCO, the "
            f"altitudes of the Sun and the Moon, and the angle from the Sun, the horizon and the Moon (when up) to g − I, the "
            f"closest one highlighted. Epochs where at least {100 * departure_fraction:.0f}% of "
            f"the exposures, and at least {departure_min}, depart are marked with ⚠ in the list and given as buttons, to find "
            f"them for further investigation.</p>"
            '<div class="eb-controls"><label>Source <select id="eb-source"></select></label>'
            '<label>Channel <select id="eb-channel"></select></label><label>Epoch <select id="eb-epoch"></select></label></div>'
            '<div class="eb-flags" id="eb-flags"></div><div class="eb-legend" id="eb-legend"></div><div class="eb-grid" id="eb-grid"></div>'
            + BROWSER_SCRIPT.replace("@@DATA@@", data_json) + "</section>")


def figure_alignment_rose(fit: Dict, bin_width: float = 20.0) -> go.Figure:
    """Direction of the measured gradients and of the gradients minus the instrument model, relative to each sky direction

    One polar histogram per sky direction (Sun, horizon, Moon when up): the
    fraction of exposures per bin of the angle from that direction to the
    gradient. A gradient that follows the sky piles up near 0°; one fixed in
    the IFU spreads as the sky direction moves across it.
    """
    data = fit["data"]
    references = [(column, name, color) for column, name, color in SKY_REFERENCES
                  if column in data.columns and np.isfinite(data[column].astype(float)).sum() >= 3]
    fig = make_subplots(rows=1, cols=max(len(references), 1), specs=[[{"type": "polar"}] * max(len(references), 1)],
                        subplot_titles=[f"from {SKY_NOUN[name]}" for _, name, _ in references] or ["no sky directions"])
    # bins centered on 0, 20, ..., 340 degrees, so a gradient along the sky direction falls in one bin
    edges = np.arange(-bin_width / 2, 360 - bin_width / 2 + 1e-9, bin_width)
    centers = edges[:-1] + bin_width / 2
    for k, (column, name, color) in enumerate(references, start=1):
        rows = data[np.isfinite(data[column].astype(float))]
        if column == "dir_moon" and "moon_alt" in rows.columns:
            rows = rows[rows["moon_alt"].astype(float) > 0]
        for label, gx, gy, style in (("g", rows["gx"], rows["gy"], dict(color=THEME["axis"])),
                                     ("g − I", rows["gx"] - rows["inst_x"], rows["gy"] - rows["inst_y"], dict(color=SERIES[0]))):
            angle = _wrap_angle(np.degrees(np.arctan2(gy, gx)) - rows[column].astype(float))
            counts, _ = np.histogram(np.mod(angle + bin_width / 2, 360) - bin_width / 2, bins=edges)
            mean, length = _resultant_length(angle)
            fig.add_trace(go.Barpolar(r=counts / max(len(rows), 1), theta=centers, width=bin_width * 0.9, name=label, legendgroup=label,
                                      showlegend=k == 1, marker=dict(color=style["color"], opacity=0.75 if label == "g" else 0.85),
                                      hovertemplate=f"{label}: %{{r:.0%}} at %{{theta:.0f}}° from {SKY_NOUN[name]}"
                                                    f"<br>R = {length:.2f}, mean {mean:.0f}° (n = {len(rows)})<extra></extra>"), row=1, col=k)
    fig.update_layout(base_layout(400, margin=dict(l=32, r=32, t=96, b=24),
                                  legend=dict(orientation="h", x=0, y=0.99, yanchor="top", yref="container")))
    # subplot titles above the 90 deg label of each rose
    fig.update_annotations(yshift=24, font=dict(size=13, color=THEME["ink"]))
    fig.update_polars(angularaxis=dict(direction="counterclockwise", rotation=0, dtick=45, tickfont=dict(size=10), gridcolor=THEME["grid"],
                                       linecolor=THEME["axis"]),
                      radialaxis=dict(tickformat=".0%", nticks=3, angle=-22.5, tickangle=-22.5, tickfont=dict(size=9, color=THEME["muted"]),
                                      gridcolor=THEME["grid"], linecolor=THEME["axis"]),
                      bgcolor=THEME["surface"])
    return fig


SLIT_PARTS = (("all", "g (pipeline plane)", THEME["muted"]), ("slit", "trends along the slits", SERIES[1]), ("ifu", "IFU part", SERIES[0]))
SLIT_SOURCE_SYMBOLS = {"twilight": "circle", "dome": "square", "science": "diamond-open"}


def _slit_exposures(slit: pd.DataFrame) -> pd.DataFrame:
    """Median of the windows or sky lines of each file of a slit trends table"""
    numeric = [c for c in slit.columns if c not in ("source", "measure", "window", "filename", "channel") and pd.api.types.is_numeric_dtype(slit[c])]
    data = slit.groupby(["source", "filename", "channel"], as_index=False)[numeric].median()
    for part in ("all", "slit", "ifu"):
        data[f"{part}_g"] = np.hypot(data[f"{part}_x"], data[f"{part}_y"])
        data[f"{part}_dir"] = np.degrees(np.arctan2(data[f"{part}_y"], data[f"{part}_x"]))
    return data.sort_values(["mjd", "expnum"]).reset_index(drop=True)


def _slit_profiles(data: pd.DataFrame, channel: str) -> Tuple[np.ndarray, np.ndarray]:
    """Trends along the slits of each exposure (%), from their coefficients and the fibermap, and the fiber index along the slits"""
    from lvmdrp.main import read_fibermap

    fibermap = read_fibermap(as_table=True)
    specid = np.asarray(fibermap["spectrographid"], dtype=int)
    upper = np.asarray(fibermap[f"ypix_{channel}"], dtype=float) >= 2040
    position = np.zeros(specid.size)
    for spid in np.unique(specid):
        select = np.where(specid == spid)[0]
        position[select] = np.linspace(-1, 1, select.size)
    degree = max(int(c.split("_")[1]) for c in data.columns if re.fullmatch(r"c\d+_\d+", c))
    profiles = np.full((len(data), specid.size), np.nan)
    for spid in np.unique(specid):
        select = specid == spid
        columns = [f"c{spid}_{k}" for k in range(degree + 1)]
        if not set(columns) <= set(data.columns):
            continue
        coeffs = data[columns].to_numpy(float)
        log_trend = coeffs[:, :1] * upper[select] + np.array([np.polynomial.legendre.legval(position[select], [0] + list(c[1:])) for c in coeffs])
        trend = np.exp(log_trend)
        profiles[:, select] = 100 * (trend / trend.mean(axis=1, keepdims=True) - 1)
    return np.arange(specid.size), profiles


def figure_slit_profiles(data: pd.DataFrame, channel: str) -> go.Figure:
    """Trends along the slits of each exposure, one panel per source, colored by night"""
    sources = [s for s in SLIT_PRODUCTS.values() if s[0] in set(data["source"])]
    fig = make_subplots(rows=len(sources), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[{"twilight": "Twilights (continuum)", "dome": "Dome flats (continuum)", "science": "Science frames (sky lines)"}[s]
                                        for s, _ in sources])
    fiber, profiles = _slit_profiles(data, channel)
    nights = data["mjd"].to_numpy(float)
    span = max(np.ptp(nights), 1.0)
    for row, (source, _) in enumerate(sources, start=1):
        for i in np.where(data["source"].to_numpy() == source)[0]:
            color = sample_colorscale(SEQUENTIAL, [0.15 + 0.85 * (nights[i] - nights.min()) / span])[0]
            r = data.iloc[i]
            fig.add_trace(go.Scatter(x=fiber, y=profiles[i], mode="lines", line=dict(width=1.4, color=color), showlegend=False,
                                     hovertemplate=f"MJD {int(r.mjd)} · exposure {int(r.expnum)}<br>fiber %{{x}}: %{{y:.2f}}%<extra></extra>"),
                          row=row, col=1)
        for edge in (648, 1296):
            fig.add_vline(x=edge - 0.5, line=dict(color=THEME["axis"], width=1), row=row, col=1)
        fig.update_yaxes(title_text="Trend (%)", row=row, col=1)
    # color bar of the nights
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=[[0, sample_colorscale(SEQUENTIAL, [0.15])[0]], [1, SEQUENTIAL[-1][1]]],
                                         cmin=nights.min(), cmax=nights.min() + span, color=[nights.min()],
                                         colorbar=dict(title=dict(text="MJD", side="right"), thickness=10, len=0.8, outlinewidth=0))),
                  row=1, col=1)
    fig.update_layout(base_layout(260 + 200 * len(sources)))
    style_axes(fig)
    fig.update_xaxes(title_text="Fiber along the slits (spectrographs 1 | 2 | 3)", row=len(sources), col=1)
    return fig


def figure_slit_split(data: pd.DataFrame, breaks: Sequence[int] = ()) -> go.Figure:
    """Strength and direction of the pipeline plane g, of its part made by the trends along the slits and of the IFU part, against the night"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    rank = data.groupby("mjd").cumcount().to_numpy()
    size = data.groupby("mjd")["mjd"].transform("size").to_numpy()
    for j, (part, label, color) in enumerate(SLIT_PARTS):
        for source, symbol in SLIT_SOURCE_SYMBOLS.items():
            sel = (data["source"] == source).to_numpy()
            if not sel.any():
                continue
            x = data["mjd"].to_numpy(float)[sel] + 0.8 * ((rank[sel] + 1) / (size[sel] + 1) - 0.5) + 1.5 * (j - 1)
            custom = np.stack([data["mjd"][sel].astype(int), data["expnum"][sel].astype(int)], axis=-1)
            common = dict(mode="markers", legendgroup=part, customdata=custom,
                          marker=dict(size=7, color=color, symbol=symbol, opacity=0.85, line=dict(width=1.2, color=color)))
            name = f"{label} · {source}"
            fig.add_trace(go.Scatter(x=x, y=data[f"{part}_g"][sel], name=name,
                                     hovertemplate=f"MJD %{{customdata[0]}} · exposure %{{customdata[1]}}<br>{label} %{{y:.2f}}%<extra></extra>",
                                     **common), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=data[f"{part}_dir"][sel], name=name, showlegend=False,
                                     hovertemplate=f"MJD %{{customdata[0]}} · exposure %{{customdata[1]}}<br>{label} towards %{{y:.0f}}°<extra></extra>",
                                     **common), row=2, col=1)
    nights = data["mjd"].to_numpy(float)
    pad = max(4.0, 0.03 * np.ptp(nights))
    lo, hi = nights.min() - pad, nights.max() + pad
    for mjd in breaks:
        if lo <= mjd <= hi:
            fig.add_vline(x=mjd, line=dict(color=THEME["muted"], width=1, dash="dot"))
    fig.update_xaxes(range=[lo, hi])
    fig.update_layout(base_layout(560))
    style_axes(fig)
    fig.update_yaxes(title_text="Strength (%)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Direction (°)", range=[-180, 180], dtick=90, row=2, col=1)
    fig.update_xaxes(title_text="MJD", row=2, col=1)
    return fig


def _slit_table(data: pd.DataFrame, breaks: Sequence[int]) -> str:
    """Median vectors of g, of the trends along the slits and of the IFU part per source and segment"""
    data = data.assign(segment=_segments(data["mjd"].to_numpy(float), breaks))
    rows = []
    for (source, _), group in data.groupby(["source", "segment"], sort=False):
        cells = [escape(source), f"{int(group.mjd.min())}–{int(group.mjd.max())}", f"{group.mjd.nunique()} / {len(group)}"]
        for part, _, _ in SLIT_PARTS:
            gx, gy = group[f"{part}_x"].median(), group[f"{part}_y"].median()
            cells.append(f"{np.hypot(gx, gy):.2f} @ {np.degrees(np.arctan2(gy, gx)):.0f}°")
        cells.append(f"{group['ifu_g'].std():.2f}" if len(group) > 1 else "")
        rows.append(cells)
    return html_table([("Source", "continuum of the flats, sky lines of the science frames"), ("Nights", "first and last night of the segment"),
                       ("Nights / exp.", "nights and exposures"), ("g (%)", "plane of the pipeline fit, median vector"),
                       ("Slit trends (%)", "part of g made by the trends along the slits, median vector"),
                       ("IFU part (%)", "g minus the slit trends, median vector"), ("Scatter (%)", "standard deviation of the strength of the IFU part")],
                      rows, numeric=[False, False, True, True, True, True, True])


def slit_sections(slit: pd.DataFrame, breaks: Sequence[int], channels: str) -> Tuple[str, Dict[str, go.Figure]]:
    """The sections of the dashboard with the trends along the slits of each channel"""
    html, figures = "", {}
    data = _slit_exposures(slit)
    for channel in [c for c in channels if c in set(data["channel"])]:
        subset = data[data["channel"] == channel].reset_index(drop=True)
        figures[f"slitsplit|{channel}"] = figure_slit_split(subset, breaks)
        figures[f"slitprofiles|{channel}"] = figure_slit_profiles(subset, channel)
        last = subset[subset["source"] == subset["source"].iloc[-1]]
        last = last[_segments(last["mjd"].to_numpy(float), breaks) == _segments(last["mjd"].to_numpy(float), breaks).max()]
        sx, sy, ix, iy = (last[c].median() for c in ("slit_x", "slit_y", "ifu_x", "ifu_y"))
        sources = ", ".join(f"{n} {s}" for s, n in subset.groupby("source")["filename"].nunique().items())
        html += (f'<details class="ig"><summary><strong>Slit trends · {channel}</strong>: last segment, slit trends {np.hypot(sx, sy):.2f}% @ '
                 f"{np.degrees(np.arctan2(sy, sx)):.0f}°, IFU part {np.hypot(ix, iy):.2f}% @ {np.degrees(np.arctan2(iy, ix)):.0f}°</summary>"
                 f'<div class="ig-body">'
                 f"<p>The throughput of the fibers changes smoothly along each slit (vignetting and the detector halves), and the "
                 f"plane that the pipeline fits across the science IFU takes up part of it, because the science fibers fill blocks of the "
                 f"slits. The sky fibers share the slits but not the science IFU, so a fit of the trends along the slits with the sky fibers "
                 f"and a plane across the science IFU tells the two apart ({escape(sources)} exposures). The trends are part of the "
                 f"throughput and belong in the flat field; only the IFU part is an illumination gradient, instrument or sky. In science "
                 f"frames the sky lines see the airglow, the same across the IFU up to a small horizon gradient, so their IFU part is the "
                 f"instrument gradient with no twilight sky.</p>"
                 f"<h3>Split of g per source and segment</h3>{_slit_table(subset, breaks)}"
                 f"<h3>Split of g against the night</h3><p>Pipeline plane g (gray), its part made by the trends along the slits "
                 f"(orange) and the IFU part (blue), for twilights (filled) and science frames (open diamonds); the dotted lines are the "
                 f"breaks.</p>"
                 f'<div class="chart" data-fig="slitsplit|{channel}"></div>'
                 f"<h3>Trends along the slits</h3><p>The fitted trend of each exposure along the three slits, relative to the mean of "
                 f"each slit, colored by night. Stable trends are a property of the fibers, which the flat field keeps.</p>"
                 f'<div class="chart" data-fig="slitprofiles|{channel}"></div>'
                 f"</div></details>")
    return html, figures


def _definitions_for(sources: Sequence[str], roses: bool, direction_scan: bool, browser: bool, slit: bool = False) -> str:
    """The definitions of the quantities shown in a run: without the cards of plots that are not on the page, and without
    the sentences about sources of exposures that are not used (tagged ``<span data-sources="...">``)"""
    sources = set(sources) | ({"joint"} if len(set(sources)) > 1 else set())
    cards = {"Direction rose": roses, "Geometry of the sky gradient": direction_scan and "twilight" in sources,
             "Maps across the IFU": browser, "Twilight I applied to other sources": "twilight" in sources and bool({"dome", "science"} & sources),
             "Trends along the slits": slit}
    definitions = _without_cards(DEFINITIONS, [title for title, shown in cards.items() if not shown])

    def keep(match):
        return match.group(2) if set(match.group(1).split()) & sources else ""

    definitions = re.sub(r'<span data-sources="([^"]+)">(.*?)</span>', keep, definitions, flags=re.S)
    return re.sub(r"[ \t]*\n[ \t]*(?=\n)", "", definitions)


def _without_cards(definitions: str, titles: Sequence[str]) -> str:
    """The definitions without the cards of the optional plots that are not shown"""
    for title in titles:
        start = definitions.index(f'    <div class="def">\n      <h3>{title}</h3>')
        end = definitions.find('    <div class="def">', start + 1)
        end = end if end >= 0 else definitions.index("  </div>\n</section>")
        definitions = definitions[:start] + definitions[end:]
    return definitions


def _comparison_table(table: pd.DataFrame) -> str:
    """Models of one channel and source; the instrument gradient of every model is in the models CSV"""
    sky = table["sky_model"].str.replace(r"^\w+: ", "", regex=True) if table["sky_model"].str.count(";").max() == 0 else table["sky_model"]
    rows = [[escape(FRAME_SHORT[r.frame]), escape(label), escape(str(r.breaks)) + (" + drift" if r.drift else ""),
             str(r.k), _num(r.rms, ".2f"), _num(r.cv_rms, ".2f"), _num(r.dbic, ".0f"), "degenerate" if r.degenerate else ""]
            for r, label in zip(table.itertuples(), sky)]
    return html_table([("Frame of I", "frame of the instrument gradient I (φ: K-mirror rotation)"), ("Sky", "direction (amplitude)"),
                       ("Breaks", "where I may jump"), ("k", "parameters"), ("rms", "fit residual r per exposure (%)"),
                       ("CV rms", "leave-one-night-out residual (%)"), ("ΔBIC", "from the best model"),
                       ("", "rank deficient, a poorly constrained segment, or a K-mirror frame with a fixed K-mirror")],
                      rows, numeric=[False, False, False, True, True, True, True, False])


def _polar_errors(gx, gy, sx, sy):
    """Errors of the strength (%) and direction (degrees) of a vector from the errors of its components"""
    strength = max(np.hypot(gx, gy), 1e-9)
    return (float(np.hypot(gx * sx, gy * sy) / strength),
            float(np.degrees(np.hypot(gy * sx, gx * sy) / strength ** 2)))


def _instrument_table(instrument: pd.DataFrame) -> str:
    rows = []
    for r in instrument.itertuples():
        jackknife = np.isfinite(r.gx_err) and np.isfinite(r.gy_err)
        sx, sy = (r.gx_err, r.gy_err) if jackknife else (r.gx_sig, r.gy_sig)
        strength_err, direction_err = _polar_errors(r.gx, r.gy, sx, sy)
        mark = "" if jackknife else "*"
        rows.append([f"{int(r.mjd_first)}–{int(r.mjd_last)}", f"{r.nights} / {r.n}", f"{r.strength:.2f} ± {strength_err:.2f}{mark}",
                     f"{r.direction:.0f} ± {min(direction_err, 180):.0f}{mark}", _num(r.phi_spread, ".1f")])
    note = "jackknife error over nights; * formal error, with fewer than 3 nights"
    return html_table([("Nights", "first and last night of the segment"), ("Nights / exp.", "nights and exposures"),
                       ("Strength (%)", note), ("Direction (°)", "from IFU +x towards +y; " + note),
                       ("K-m. (°)", "spread of the K-mirror rotation; near 0 the K-mirror frames equal the IFU")],
                      rows, numeric=[False, True, True, True, True])


def _alignment_table(alignment: pd.DataFrame) -> str:
    rows = [[escape(r.gradient), escape(r.frame), _num(r.R, ".2f"), _num(r.mean, ".0f"), _num(r.p, ".1e"), str(r.n)] for r in alignment.itertuples()]
    return html_table([("Gradient", "g: measured; g − I: minus the instrument gradient"), ("Relative to", "direction measured in the IFU or from a sky direction"),
                       ("R", "mean resultant length"), ("Mean (°)", "circular mean"), ("p", "Rayleigh test"), ("n", "")],
                      rows, numeric=[False, False, True, True, True, True])


def qa_instrument_gradient(twilight=None, dome=None, science=None, slit=None, breaks: Sequence[int] = None, channels: str = "brz",
                           kmirror=None, skies: Dict[str, Dict] = None, epochs: Dict[int, Dict] = None, roses: bool = False,
                           direction_scan: bool = False, output_dir: str = None, name: str = "instrument_gradient") -> Dict:
    """Tests for an instrument gradient across the science IFU and writes a dashboard

    For each channel and source (twilight, dome, science, and all of them
    together), fits the gradient model of the module docstring with every
    instrument frame (none, IFU, before the K-mirror) and with and without the
    breaks, and ranks the models by BIC. For the best model with an
    instrument gradient it shows the instrument gradient per segment, the
    instrument gradient fitted night by night, the alignment of the residual
    gradients with the sky directions, a scan of the sky direction (twilights)
    and how well the twilight instrument gradient explains the dome and
    science gradients.

    Parameters
    ----------
    twilight, dome : str or pandas.DataFrame, optional
        Tables of :func:`lvmdrp.qa.fiberflats.qa_twilight_trends` and
        :func:`~lvmdrp.qa.fiberflats.qa_dome_trends`, or their CSV paths.
    science : str or pandas.DataFrame, optional
        Table of :func:`science_gradients`, or its CSV path.
    slit : str or pandas.DataFrame, optional
        Table of :func:`slit_trend_split`, or its CSV path, for the sections of
        the trends along the slits. Default is none.
    breaks : list[int], optional
        MJDs where the instrument gradient may jump, e.g. from
        :func:`intervention_breaks`. Default is none.
    channels : str, optional
        Channels to fit. Default is 'brz'.
    kmirror : dict or str, optional
        K-mirror relation, for the sign of the field angle.
    skies : dict, optional
        Sky models to compare, {label: {source: {direction column: amplitude}}}.
        Default is :data:`DEFAULT_SKY` and, as an alternative, a free amplitude
        per night along the same directions.
    output_dir : str, optional
        Directory of the dashboard and tables. Default is
        ``qa/fiberflat/{name}`` in the master calibrations directory.
    epochs : dict, optional
        Calibration epochs, to group the twilight and dome flat exposures of the
        maps by epoch. Default is the epochs file in lvmcore; without it, they
        are grouped by segment.
    roses : bool, optional
        Add the polar histograms of the direction of g and g - I relative to
        each sky direction (see :func:`figure_alignment_rose`). Default is False.
    direction_scan : bool, optional
        Refit the twilights with the direction of the Sun rotated and mirrored
        and plot the rms against the offset (see :func:`sky_direction_scan`),
        a test of the geometry. Default is False.
    name : str, optional
        Name of the dashboard. Default is 'instrument_gradient'.

    Returns
    -------
    dict
        comparison (pandas.DataFrame of all the models), instrument
        (pandas.DataFrame of the instrument gradient of the best model per
        channel and source), cross_checks, figures and report (path).
    """
    inputs = {"twilight": twilight, "dome": dome, "science": science}
    sources = {source: prepare_gradients(table, source, kmirror=kmirror, channels=channels)
               for source, table in inputs.items() if table is not None}
    slit = pd.read_csv(slit) if isinstance(slit, (str, os.PathLike)) else slit
    if not sources and (slit is None or slit.empty):
        raise ValueError("give at least one table of gradients (twilight, dome or science) or of slit trends")
    breaks = sorted(int(b) for b in (breaks or []))
    skies = skies or {"default": DEFAULT_SKY,
                      "per night": {s: {column: "night" for column in terms} for s, terms in DEFAULT_SKY.items()}}
    breaks_options = {"none": []} if not breaks else {"none": [], "interventions": breaks}
    output_dir = output_dir or sandbox_qa_dir("fiberflat", name)
    os.makedirs(output_dir, exist_ok=True)

    groups = dict(sources)
    if len(sources) > 1:
        groups["joint"] = pd.concat(list(sources.values()), ignore_index=True)

    comparisons, instruments, figures, sections, cross_checks = [], [], {}, [], {}
    positions = ifu_fiber_positions()
    browser_fits = {}
    if epochs is None:
        try:
            epochs = load_calibration_epochs()
        except Exception as e:
            log.warning(f"cannot read the calibration epochs ({type(e).__name__}: {e}), grouping the maps by segment")
    for group, data in groups.items():
        for channel in [c for c in channels if c in set(data["channel"])]:
            subset = data[data["channel"] == channel].reset_index(drop=True)
            table, fits = compare_gradient_models(subset, skies=skies, breaks=breaks_options)
            table.insert(0, "channel", channel)
            table.insert(0, "group", group)
            comparisons.append(table)
            best = table[(table["frame"] != "none") & ~table["degenerate"] & np.isfinite(table["bic"])].head(1)
            if best.empty:
                log.warning(f"no constrained model with an instrument gradient for the {group} gradients of channel {channel} ({len(subset)} exposures)")
                continue
            best = best.iloc[0]
            fit = fits[(best.frame, best.sky, best.breaks, best.drift)]
            instrument = fit["instrument"].copy()
            instrument.insert(0, "frame", best.frame)
            instrument.insert(0, "channel", channel)
            instrument.insert(0, "group", group)
            instruments.append(instrument)
            if group in BROWSER_SOURCES:
                browser_fits[(group, channel)] = fit
            sky_model = skies[best.sky]
            nightly_fit = fit_gradient_model(subset, frame="ifu", sky=sky_model, breaks="nights")
            nightly, nightly_note = _nightly_or_best(nightly_fit, fit)
            figures[f"vectors|{group}|{channel}"] = figure_vectors(fit)
            figures[f"nightly|{group}|{channel}"] = figure_nightly_instrument(nightly, breaks)
            figures[f"residuals|{group}|{channel}"] = figure_residuals(fit)
            if roses:
                figures[f"rose|{group}|{channel}"] = figure_alignment_rose(fit)
            alignment = residual_alignment(fit)

            scan_html = ""
            if group == "twilight" and direction_scan:
                column, amplitude = next(iter(DEFAULT_SKY["twilight"].items()))
                scan = sky_direction_scan(subset, column=column, amplitude=amplitude, breaks=fit["breaks"])
                figures[f"scan|{group}|{channel}"] = figure_direction_scan(scan)
                best_scan = scan.loc[scan["bic_inst"].idxmin()]
                scan_html = (f"<h3>Geometry of the sky gradient</h3><p>The fit with the direction of the Sun rotated by an offset, "
                             f"and mirrored. The best offset is {best_scan.offset:+.0f}°{' mirrored' if best_scan.mirrored else ''} "
                             f"(rms {best_scan.rms_inst:.2f}% against {scan[(scan.offset == 0) & ~scan.mirrored].rms_inst.iloc[0]:.2f}% "
                             f"with no offset); without an instrument gradient the best rms is {scan.rms_sky.min():.2f}%.</p>"
                             f'<div class="chart" data-fig="scan|{group}|{channel}"></div>')

            check_html = ""
            if group == "twilight":
                for other in ("dome", "science"):
                    if other not in sources:
                        continue
                    other_data = sources[other][sources[other]["channel"] == channel].reset_index(drop=True)
                    if other_data.empty:
                        continue
                    check = cross_check(other_data, fit, DEFAULT_SKY[other])
                    cross_checks[(other, channel)] = check
                    if check["n"] >= 3:
                        check_html += (f"<h3>Twilight I applied to the {other} gradients</h3><p>{check['n']} exposures: rms of the "
                                       f"gradients {check['raw']:.2f}% before and {check['corrected']:.2f}% after removing the twilight "
                                       f"instrument gradient; rms of their sky-only fit {check['sky_raw']:.2f}% before and "
                                       f"{check['sky_corrected']:.2f}% after.</p>" + _alignment_table(check["alignment"]))

            last = fit["instrument"].iloc[-1]
            sections.append(
                f'<details class="ig"{" open" if not sections else ""}><summary><strong>{escape(group.capitalize())} · {channel}</strong>: '
                f'I {escape(INSTRUMENT_FRAMES[best.frame][0])}, last segment {last.strength:.2f}% @ {last.direction:.0f}°</summary>'
                f'<div class="ig-body">'
                f"<h3>Model comparison</h3>{_comparison_table(table)}"
                f"<h3>Instrument gradient I of the best model ({escape(INSTRUMENT_FRAMES[best.frame][0])}, sky {escape(best.sky)}, "
                f"breaks {escape(str(best.breaks))})</h3>"
                f'<div class="ig-pair">{_instrument_table(fit["instrument"])}<div class="chart" data-fig="vectors|{group}|{channel}"></div></div>'
                f"<h3>Instrument gradient I night by night</h3><p>One instrument gradient I fixed in the IFU per night (filled) with the sky "
                f"model shared by all the nights; a jump marks a change of the instrument, the dotted lines are the breaks. For each "
                f"exposure, spread within its night: the measured gradient (g, gray dots), g minus the I of its night (g − I, open "
                f"circles) and, in the direction panel, the direction of its sky gradient (crosses: {_sky_nouns(subset)}). A sky "
                f"gradient left in g − I follows the crosses.{nightly_note}</p>"
                f'<div class="chart" data-fig="nightly|{group}|{channel}"></div>'
                f"<h3>Direction of g and g − I</h3><p>After a correct instrument gradient I is removed, g − I keeps its "
                f"direction relative to the sky (R near 1) instead of the IFU. The plot below gives the fit residual of each exposure, "
                f"|r| = |g − I − sky|, against the night, to spot nights the model does not describe.</p>"
                f'{_alignment_table(alignment)}<div class="chart" data-fig="residuals|{group}|{channel}"></div>'
                + (f"<p>Angle from each sky direction to the measured gradient (g, gray) and to the gradient minus the instrument model "
                   f"(g − I, blue), as the fraction of exposures per 20° bin. If what is left after removing the instrument gradient is a "
                   f"sky gradient, g − I piles up near 0° for its sky direction: the Sun in twilights, the horizon (airglow) or the Moon "
                   f"in science frames, the horizon for the dome screen.</p>"
                   f'<div class="chart" data-fig="rose|{group}|{channel}"></div>' if roses else "")
                + f"{scan_html}{check_html}</div></details>")

    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    instrument = pd.concat(instruments, ignore_index=True) if instruments else pd.DataFrame()
    comparison.to_csv(os.path.join(output_dir, f"{name}_models.csv"), index=False)
    instrument.to_csv(os.path.join(output_dir, f"{name}_instrument.csv"), index=False)

    tiles = []
    for (group, channel), row in (instrument.groupby(["group", "channel"]).last().iterrows() if not instrument.empty else []):
        rows = comparison[(comparison["group"] == group) & (comparison["channel"] == channel)]
        none = rows[rows["frame"] == "none"]["bic"].min()
        best_bic = rows[(rows["frame"] != "none") & ~rows["degenerate"]]["bic"].min()
        tiles.append((f"I · {group.capitalize()} · {channel}", f"{row.strength:.2f}% @ {row.direction:.0f}°",
                      f"last segment, {INSTRUMENT_FRAMES[row.frame][0]}; ΔBIC {none - best_bic:.0f} over no I"))
    meta = [("Sources", ", ".join(f"{s}: {d['expnum'].nunique()} exposures, {d['mjd'].nunique()} nights" for s, d in sources.items())),
            *([("Slit trends", ", ".join(f"{source}: {n} files" for source, n in slit.groupby("source")["filename"].nunique().items()))]
              if slit is not None and not slit.empty else []),
            ("Channels", channels), ("Breaks", _label_breaks(breaks)),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))]
    intro = (_introduction(list(sources)) if sources else
             "<p>Splits the gradient that the pipeline fits across the science IFU into the smooth throughput trends along the "
             "slits, measured with the sky fibers, and the IFU part, the illumination gradient. Every quantity is defined under "
             '<a href="#definitions">How the quantities are computed</a>.</p>')
    browser = epoch_browser(browser_fits, positions, epochs)
    slit_html = ""
    if slit is not None and not slit.empty:
        slit_html, slit_figures = slit_sections(slit, breaks, channels)
        figures.update(slit_figures)
    body = (PAGE_STYLE + ("".join(sections) or ("<p>No model with an instrument gradient could be fitted.</p>" if sources else ""))
            + slit_html + browser + _definitions_for(list(sources), roses, direction_scan, bool(browser), bool(slit_html)))
    report = write_dashboard(os.path.join(output_dir, f"{name}.html"), "LVM Instrument Gradient", "LVM DRP · fiber flats · IFU gradient",
                             intro, meta, tiles, body, figures, "lvmdrp.qa.ifu_gradient.qa_instrument_gradient")
    log.info(f"written instrument gradient dashboard to {report}")
    return {"comparison": comparison, "instrument": instrument, "cross_checks": cross_checks, "figures": figures, "report": report}

