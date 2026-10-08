#!/usr/bin/env python3

"""
Run sky spectral decomposition on a median-stacked LVM frame.

Usage:
    python decompose_parallel.py <data_file> [palace_dir] [options]

Example:
    python decompose_parallel.py lvmsframe_median_stack.fits ../ --n-workers 8
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# clarabel (Rust QP solver) and any other Rust/Rayon library ignore OMP_NUM_THREADS.
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import queue as queue_mod
import sys
import time
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp

import numpy as np
from astropy.io import fits
from tqdm import tqdm

# ``python /path/to/skysub/decompose_parallel.py`` puts only ``skysub/`` on
# sys.path.  Add the repository/package root so direct-script and ``-m``
# execution use the same fully-qualified package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skysub.sky_decomp.result_io import results_to_fits
from skysub.sky_decomp.moon_zodi_model import (
    DEFAULT_DATA_ROOT as DEFAULT_MOON_ZODI_DATA_ROOT,
    SKYFAR_LINEAR_RIDGE_LAMBDA,
    SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX,
    file_sha256,
    validate_decomposition_data_root,
    wave_sha256,
)
from skysub.sky_decomp.fit import SPLIT_ZODI_CONTINUUM_DEFAULTS

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # threadpoolctl is optional; env vars are the fallback.
    threadpool_limits = None


def _clamp_native_threads(n=1):
    """Force every loaded thread pool (BLAS/OpenMP/Rayon/TBB/etc.) to `n` threads."""
    # Redundant with the env vars but catches lazy imports and fork-inherited pools.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "NUMBA_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        os.environ[var] = str(n)
    if threadpool_limits is not None:
        threadpool_limits(limits=n)


_clamp_native_threads(1)


_WORKER_DECOMPOSER = None
_WORKER_FACTOR = 1.0
_WORKER_HDU = None
_WORKER_FLUX = {}
_WORKER_LSF = {}
_WORKER_META = None
_WORKER_PROGRESS_QUEUE = None
_WORKER_FIT_MODEL = "baseline"
_WORKER_EXPOSURE_SECONDS = 900.0
_WORKER_WAVE = None
_WORKER_LOG_WAVE = None  # log(wave), for the reversal colour test
_WORKER_REVERSAL_RETRY_BOUND = None  # None disables retry-on-reversal
_WORKER_TELLURIC_CALCULATOR = None
_WORKER_DECOMPOSER_KWARGS = {}
_WORKER_SCIENCE_LINE_MASK = None
_WORKER_SCIENCE_LINE_FWHM = None
_WORKER_SCIENCE_LINE_CENTRE = True  # overwritten by init_worker
_WORKER_FIT_PIXEL_WEIGHTS = False  # overwritten by init_worker
_WORKER_FIT_PIXEL_WEIGHT_CLIP = None  # overwritten by init_worker
_WORKER_SENS_ABS = None  # absolute sensitivity on _WORKER_WAVE, when weighting
_WORKER_COMPACT_CACHE_DIR = None
_WORKER_RUN_FINGERPRINT = None
_PWV_FALLBACK_REPORTED = False

# Match lvmdrp.functions.fluxCalMethod.DEFAULT_PWV for invalid or missing PWV_MED.
DRP_DEFAULT_PWV_MM = 15.0

# --- per-pixel fit weights -------------------------------------------------
# ON BY DEFAULT since 2026-09-18; `--no-fit-pixel-weights` restores the old
# behaviour.
#
# The fit USED to be run with ivar = 1 on every pixel -- an unweighted MASK with
# 0 only in the science-line windows, not a variance, and not even
# isfinite-filtered.  Every pixel counted equally, from the OH band heads to the
# faint inter-band continuum, although their photon noise differs by more than an
# order of magnitude.  The ML loss has used the absolute photon model since
# 2026-09-09; this puts the FIT on the same footing, sharing the curves and the
# variance floor through `sky_decomp.pixel_weights` so the two cannot drift apart.
#
# The weights are row-normalised to mean 1 over the good pixels, which means
# exptime, the fibre count and FACTOR all cancel: only the SHAPE of
# flux*sens(lambda) survives, plus the 5%-of-median variance floor.  That is
# deliberate -- it keeps the overall scale exactly where the unweighted mask had
# it, so `moon_smooth_lambda`, `zodi_smooth_lambda`, the LSF `roughness_fraction`
# and `line_weight`, all tuned against ivar = 1, keep their meaning.
#
# MEASURED on the full 1447-row every10 telluric subset against the unweighted
# products for the same rows, scored on the ABSOLUTE single-fibre photon chi2
# (`reduced_chi2` becomes a weighted chi2 and its values stop being comparable
# across this change -- judge runs on the absolute metric instead):
#
#     arm    full band     blue      rows improved
#     sci      -11.6%     -1.9%       1421/1445
#     near     -11.2%     -3.1%       1444/1445
#     far      -14.0%     -3.7%       1438/1444
#
# The blue improves on every arm, so the fit is NOT trading it away for OH --
# which is what the 3x red/blue weight ratio made me expect.  The cost is a
# PARTITION one that chi2 cannot see: the deployed diffuse-zeroed gate
# (mlp_predictor.data.diffuse_zeroed_mask) goes 42 -> 51 rows, 12 newly collapsed
# against 3 recovered, i.e. -0.62% yield.  The diffuse block moves >0.1 dex on
# 5.5% of rows and >0.3 dex on 2.6%; OH, moon and zodi barely move.  Fit status
# is unchanged.
#
# Confirmed on a 1000-row run with retry-on-reversal also enabled: -11.1% sci /
# -10.9% near / -14.8% far full band, blue -2.5/-1.8/-4.4%, fit status identical
# to baseline.  NOTE the two changes interact -- the weights RAISE the first-fit
# reversal count (sci 13 -> 17) by moving the moon/zodi partition, and the retry
# absorbs it (43/43 recovered).  Turning the weights on WITHOUT the retry would
# cost yield rather than save it.
FIT_PIXEL_WEIGHTS = True
# Bound on the weight dynamic range, as a factor about the row mean: `3.0` keeps
# every weight inside [1/3, 3].  None leaves the raw photon weights, which span
# ~600x within a row.  A 20-row probe had put the diffuse-collapse rate at 5%
# and clipping at 3 removed it for half the chi2 gain; on the full 1447-row
# subset the true rate is 0.83%, so the clip is NOT needed and defaults to off.
FIT_PIXEL_WEIGHT_CLIP = None

FIT_PIXEL_WEIGHT_FLOOR_FRAC = 0.05
FIT_PIXEL_WEIGHT_EXPTIME_S = 900.0
# Per-arm fibre-count column; only used when the weights are NOT normalised.
FIT_PIXEL_WEIGHT_FIBRE_COLUMN = {
    "sci": "fibers_sci_used",
    "sky1": "fibers_sky_near_used",
    "sky2": "fibers_sky_far_used",
}


class _FailedInputError(ValueError):
    """Known bad input that must fail only its own row, not the whole chunk.

    Every subclass is caught by `_fit_worker_row` and turned into a
    same-schema NaN row through `failed_input_result`, so the row is recorded
    with `fit_status=failed_input` and its reason instead of aborting the
    worker.
    """


class _InvalidAirmassError(_FailedInputError):
    """Known bad-coordinate input that should fail only its own row."""


class _UnusableLSFError(_FailedInputError):
    """A row whose detector LSF has no pixel `_sanitised_lsf_row` can use."""

FIT_MODEL_SUFFIXES = {
    "baseline": "",
    "lsf-surface-iterative": "_lsf_surface_iterative",
    "lsf-surface-iterative-split-zodi": "_lsf_surface_iterative_split_zodi",
    "lsf-spline2d-split-zodi": "_lsf_spline2d_split_zodi",
    "moon-zodi-lsf-surface-iterative": "_moon_zodi_lsf_surface_iterative",
    "adam25k-telluric-lsf-spline2d": "_adam25k_telluric_lsf_spline2d",
    "palace-aijc-vnf-line-amplitude-pca30": "_palace_aijc_vnf_line_amplitude_pca30",
    "adam25k-telluric-split-zodi-lsf-spline2d": (
        "_adam25k_telluric_split_zodi_lsf_spline2d"
    ),
    "palace-aijc-vnf-split-zodi-lsf-spline2d": (
        "_palace_aijc_vnf_split_zodi_lsf_spline2d"
    ),
    "palacecorr-aijc-vnf-split-zodi-lsf-spline2d": (
        "_palacecorr_aijc_vnf_split_zodi_lsf_spline2d"
    ),
    "palace-aijc-vnf-pca30-split-zodi-lsf-spline2d": (
        "_palace_aijc_vnf_pca30_split_zodi_lsf_spline2d"
    ),
    # Compatibility aliases for existing commands and persisted provenance.
    "adam25k-telluric-niv-continuum": "_adam25k_telluric_niv_continuum",
    "palace-aijc-vnf-pca30-niv-continuum": "_palace_aijc_vnf_pca30_niv_continuum",
}
MOON_ZODI_FIT_MODEL = "moon-zodi-lsf-surface-iterative"
SPLIT_ZODI_FIT_MODEL = "lsf-surface-iterative-split-zodi"
SPLINE2D_SPLIT_ZODI_FIT_MODEL = "lsf-spline2d-split-zodi"
SPLIT_ZODI_FIT_MODELS = (SPLIT_ZODI_FIT_MODEL, SPLINE2D_SPLIT_ZODI_FIT_MODEL)
ADAM25K_TELLURIC_FIT_MODEL = "adam25k-telluric-lsf-spline2d"
PALACE_VNF_PCA30_FIT_MODEL = "palace-aijc-vnf-line-amplitude-pca30"
ADAM25K_SPLIT_ZODI_FIT_MODEL = "adam25k-telluric-split-zodi-lsf-spline2d"
PALACE_VNF_SPLIT_ZODI_FIT_MODEL = "palace-aijc-vnf-split-zodi-lsf-spline2d"
PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL = (
    "palacecorr-aijc-vnf-split-zodi-lsf-spline2d"
)
PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODEL = (
    "palace-aijc-vnf-pca30-split-zodi-lsf-spline2d"
)
LEGACY_ADAM25K_SPLIT_ZODI_FIT_MODEL = "adam25k-telluric-niv-continuum"
LEGACY_PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODEL = (
    "palace-aijc-vnf-pca30-niv-continuum"
)
# Public compatibility names used by already-generated notebooks.
ADAM25K_NIV_CONTINUUM_FIT_MODEL = LEGACY_ADAM25K_SPLIT_ZODI_FIT_MODEL
PALACE_VNF_PCA30_NIV_CONTINUUM_FIT_MODEL = (
    LEGACY_PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODEL
)
ADAM25K_SPLIT_ZODI_FIT_MODELS = (
    ADAM25K_SPLIT_ZODI_FIT_MODEL,
    LEGACY_ADAM25K_SPLIT_ZODI_FIT_MODEL,
)
PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODELS = (
    PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODEL,
    LEGACY_PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODEL,
)
SPLIT_ZODI_TELLURIC_FIT_MODELS = (
    *ADAM25K_SPLIT_ZODI_FIT_MODELS,
    PALACE_VNF_SPLIT_ZODI_FIT_MODEL,
    PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL,
    *PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODELS,
)
TELLURIC_FIT_MODELS = (
    ADAM25K_TELLURIC_FIT_MODEL,
    PALACE_VNF_PCA30_FIT_MODEL,
    *SPLIT_ZODI_TELLURIC_FIT_MODELS,
)


def _resolved_palace_oh_suffix(fit_model, palace_oh_suffix):
    if fit_model != PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL:
        return palace_oh_suffix
    if palace_oh_suffix not in (None, SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX):
        raise ValueError(
            "palacecorr requires pmd_popmodel_OH"
            f"{SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX}.dat"
        )
    return SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX


def _fit_model_primary_meta(fit_model, palace_oh_suffix):
    metadata = {
        "DECOMPM": fit_model,
        "ZODICORR": (SPLIT_ZODI_ZODI_CORRECTION, "Leinert zodi correction used by the anchor"),
    }
    if palace_oh_suffix is not None:
        metadata["OHFILE"] = f"pmd_popmodel_OH{palace_oh_suffix}.dat"
    if fit_model == PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL:
        metadata["OHRIDGE"] = SKYFAR_LINEAR_RIDGE_LAMBDA
    return metadata


# Defaults for the SkyDecompLSFSurfaceIterative(split_zodi=True) knobs; match the
# settings validated on the p40_p70 every10 identifiability notebook.
SPLIT_ZODI_N_KNOTS_DEFAULT = SPLIT_ZODI_CONTINUUM_DEFAULTS["n_zodi_spline_knots"]
SPLIT_ZODI_SMOOTH_LAMBDA_DEFAULT = SPLIT_ZODI_CONTINUUM_DEFAULTS["zodi_smooth_lambda"]
MOON_SMOOTH_LAMBDA_DEFAULT = SPLIT_ZODI_CONTINUUM_DEFAULTS["moon_smooth_lambda"]
SPLIT_ZODI_MOON_ALBEDO_PHASE_DEG = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "moon_albedo_fiducial_phase_deg"
]
SPLIT_ZODI_COLOR_EXPONENT = SPLIT_ZODI_CONTINUUM_DEFAULTS["zodi_color_exponent"]
# --- split-zodi identifiability defaults (validated 2026-09-03) --------------
# Without these the split is degenerate in a way fit quality cannot see: on 200
# lunation-stratified sky spectra the deployed configuration put the moon and
# zodi colours in the WRONG ORDER on 164 of 168 moon-up spectra (fitted moon
# log-log slope +0.21 where physics says -3.7, zodi -4.04 where physics says
# -0.3), left the moon family holding 45% of the continuum with the moon 37 deg
# BELOW the horizon, and had the fitted zodi tracking lunar illumination at
# rho = +0.95 while retaining only rho = 0.09 of its Leinert B500 dependence.
#
# The four knobs below fix those, and they are not interchangeable:
#   * the RATIO BOUNDS fix the colour ordering (reversals 164 -> 2 of 168) and
#     stop a spline zeroing out mid-band, which is how the old fit reached a
#     lower rms -- it used the moon family as piecewise scratch space;
#   * the FRACTION bracket fixes dark time (moon share 0.45 -> 0.02);
#   * the absolute LEINERT ANCHOR fixes the amplitude geometry
#     (rho(zodi, B500) 0.09 -> 0.90, rho(zodi, FLI) 0.94 -> 0.12).
# Dropping the anchor keeps the reversals fixed but leaves the zodi amplitude
# tracking the moon; dropping the bounds re-opens the reversals.
#
# Cost: median rms x1.01 over the sample, concentrated entirely at bright moon
# (x1.49 median for FLI > 0.8, ~1% of the continuum, blue-weighted).  Part of
# that is the baseline overfitting via the spline hole described above.
SPLIT_ZODI_MOON_RATIO_BOUND = SPLIT_ZODI_CONTINUUM_DEFAULTS["moon_ratio_bound"]
SPLIT_ZODI_ZODI_RATIO_BOUND = SPLIT_ZODI_CONTINUUM_DEFAULTS["zodi_ratio_bound"]

# RETRY-ON-REVERSAL.  A reversal -- fitted moon continuum REDDER than the fitted
# zodi, i.e. the two families have swapped roles -- is a MISLABELLED row, not a
# hard one, so the corpus build has always simply dropped it (2.3% of every10
# rows; 2.1% marginal past every other gate, ~220 rows on the full corpus).
# Production cannot drop a row: it still owes a prediction.  And the rows are
# recoverable, because a reversal is a SHAPE-labelling artefact rather than a
# brightness error -- measured on the 18 reversed sci rows of
# gaia-stars-mask-telluric every10, refitting with a tighter moon/zodi spline
# shape bound (moon_ratio_bound = zodi_ratio_bound):
#
#     bound   un-reversed   median chi2 ratio vs deployed 0.70
#     0.70       1/18            1.000   (control: reproduces the gate)
#     0.85      18/18            1.028
#     0.95      18/18            1.084
#
# All 18 recover at 0.85 for +2.8% median chi2 (worst single row +109%), and the
# AMPLITUDES barely move -- moon share 0.434 -> 0.433, 0.568 -> 0.568 on typical
# rows -- because the data barely distinguish the two branches
# (rho(moon, zodi) = -0.948).  Only the shape assignment flips.
#
# Retrying per row is strictly better than tightening globally: the deployed 0.70
# is kept for the 97.9% of rows that are fine and only the failures are
# tightened.  The retry is recorded in the `reliability` column either way, so a
# consumer can still tell a retried row from a clean one, and a row that stays
# reversed is flagged rather than silently written.
SPLIT_ZODI_REVERSAL_RETRY = True
SPLIT_ZODI_REVERSAL_RETRY_BOUND = 0.85
# Test geometry, matching mlp_predictor.data.split_zodi_reversal_diagnostics and
# sky_decomp.reliability: only rows whose moon share of the moon+zodi continuum
# lies inside [frac, 1 - frac] carry ordering information.
SPLIT_ZODI_REVERSAL_MIN_COMPONENT_FRAC = 0.05
SPLIT_ZODI_REVERSAL_MIN_SEPARATION = 0.0
SPLIT_ZODI_AMP_PRIOR_TOL = SPLIT_ZODI_CONTINUUM_DEFAULTS["amp_prior_tol"]
SPLIT_ZODI_ZODI_AMP_BOUND = SPLIT_ZODI_CONTINUUM_DEFAULTS["zodi_amp_bound"]

# Diffuse species-ratio bracket, ON by default since 2026-09-10.  The three
# diffuse species are individually unidentifiable in the LVM band: on the
# canonical-PALACE basis the free fit spreads log10(FeO/HO2) over 10.2 dex and
# zeroes HO2 on 23% of rows, and the THREE ARMS OF ONE EXPOSURE -- same sky, a
# few degrees apart -- disagree by 0.633 dex at the median.  Airglow does not
# vary by x4 in a species ratio over 5 deg, so that spread is fitting noise,
# and in dark time the amplitude it moves leaks straight into the zodi.
#
# Validated on 500 rows stratified over moon state x |ecl_beta|, two full
# runs: FeO/HO2 5-95 span 10.25 -> 0.40 dex, HO2 near-zero rows 23.2% -> 0.2%,
# for +2.46% of sci blue chi2 (+1.60% near, +0.13% far) and no measurable
# full-band cost.  It moves the SPLIT, not the sum: per-row log10(prior/free)
# on sci is zodi p90 0.029, diffuse p90 0.117, zodi+diffuse SUM p90 0.012.
# Bright-moon rows do not move at all -- their zodi is pinned at the Leinert
# ceiling -- so the bracket acts only where the diffuse dominates.
#
# NOMINAL: FLUX shares (HO2, FeO, O2Ac), the geometric median of the fitted
# corpus, NOT PALACE's own reference shares -- centring on PALACE costs 25.7%
# of the blue chi2 against 0.67% for the corpus median, because PALACE is
# calibrated for Cerro Paranal and LVM observes from LCO.  RE-MEASURE THIS on
# the corpus being fitted if the basis or the corpus changes; the value below
# was measured on 60 every10 rows of gaia-stars-mask on the _canonhyb_v1
# basis.
#
# KNOWN FAILURE MODE: with c >= 0 a ratio bound makes the block all-positive
# or all-zero, so a row whose fit wants HO2 = 0 loses its entire diffuse
# block (measured 0.2-0.4% of rows).  `diffuse_zeroed_keep_mask` catches
# those downstream; count them after a run.
SPLIT_ZODI_DIFFUSE_RATIO_BOUND_DEX = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_ratio_bound_dex"
]
SPLIT_ZODI_DIFFUSE_RATIO_NOMINAL = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_ratio_nominal"
]

# Moon-gated upper bound on the DIFFUSE BLOCK relative to OH.
# The three diffuse species are mesospheric chemiluminescence and cannot
# depend on the moon, yet on gaia-stars-mask-cont their amplitude relative to
# OH rises with moon_frac_po -- rho +0.716 for the block (FeO alone +0.693,
# and HO2/O2Ac inherit it through the species-ratio bracket) -- while airmass
# and van Rhijn give +0.015.  The templates are absorbing scattered
# moonlight: ~74% of the fitted FeO on full-moon rows, 3-5% of the fitted
# continuum.  OH itself is moon-independent (rho +0.088), so it is a clean
# normaliser.
#
# SCOPE IS THE BLOCK, not FeO alone.  An FeO-only cap was implemented and
# validated first; it worked (chi2-free, 99% of the released flux to the
# moon) but on ~40% of gated rows the +/-0.2 dex species-ratio bracket became
# the binding constraint instead -- log10(FeO/HO2) lower-edge occupancy rose
# 25.8% -> 40.5% -- so FeO could not fall further and the Noll anchor barely
# moved.  Capping the block lets the ratio bracket distribute the reduction
# rather than block it.  The block ratio is also better behaved: dark-time
# robust sigma 0.216 dex against 0.306 for FeO, and Theil-Sen slope against
# OH +1.16 against +1.40.
#
# CENTRE is the dark-time median of log10(A_diffuse/A_OH) over diffuse-live
# rows, where there is no moon to leak.  Re-measure it per corpus.
#
# ONE-SIDED AND GATED, both forced by measurement: the dark-time scatter is
# real -- clipping it cost 15-32% of the blue chi2 in the FeO-only sizing
# test -- so a two-sided or ungated bound is unaffordable.
#
# RELAX = 0, i.e. a flat width above the gate.  The ramp was there to avoid a
# discontinuity at the gate, but it is unnecessary: the measured excess is
# only -0.04 to +0.04 dex just above the gate, well inside the allowed 0.25,
# so the bound is naturally inactive there and turns on where the excess
# exceeds the legitimate scatter.  Median excess by moon_frac_po:
# +0.04 dex at 0.61-0.82, +0.22 at 0.82-0.91, +0.37 at 0.91-0.95,
# +0.52 at 0.95-1.00.
#
# BOUND TIGHTENED 0.25 -> 0.15 dex on 2026-09-11.  At 0.25 the bound sat at a
# ratio of 0.3991, ABOVE the observed 0.244 and 0.374 in the two lower gated
# bins, so the real moon dependence over moon_frac_po 0.6-0.9 went untouched
# and the residual correlation stayed at +0.644.  0.15 puts the bound at
# 0.317, which bites in those bins too.  Affordable because the cap measured
# chi2-FREE at 0.25: blue -0.21% on binding rows, 0.00% elsewhere, zero dark
# rows touched, collapse rate unchanged at 2.40%.
#
# BOUND SET BACK TO 0.15 on 2026-09-12, after a round trip to 0.30 and back.
# Read this before changing it again.
#
# The 0.15 -> 0.30 change was made to buy back an apparent OH regression in the
# `naive_baseline` mesospheric row (gain +1.3% -> -8.3%, and GROUP-EQUAL
# +3.0% -> -5.1%, both reading "LOSES").  THAT REGRESSION IS A METRIC
# ARTEFACT.  The mesospheric coefficient sRMSE weights all 357 OH sticks
# equally, but the OH block is internally degenerate -- neighbouring sticks
# trade amplitude with almost no change to the convolved spectrum -- so the
# same errors projected through the basis tell the opposite story:
#
#   corpus                 mesospheric COEF gain     OH FLUX gain
#   gaia-stars-mask-cont   (no cap)      +1.2%          +5.5%
#   gaia-stars-mask-cont2  (W=0.15)      -3.9%          +4.9%
#   gaia-stars-mask-cont3  (W=0.30)      -3.6%          +5.0%
#
# The coefficient gain swings 4.8 pp and twice goes negative; the flux gain is
# flat and always positive.  Every flux-space and amplitude metric was flat or
# improving across all three: reconstruction chi2 3.984/3.889/3.865, ratio to
# the decomposition's own self-fit floor 1.09/1.09/1.08, blue ratio
# 1.27/1.26/1.20, OH total amplitude MAD 0.00659/0.00648/0.00636.
# `naive_baseline` now prints a flux-space mesospheric companion and shouts
# when the two disagree in sign; use it.
#
# What 0.30 actually cost: the leak it exists to remove came back.  FeO/OH by
# moon-amplitude quartile Q4/Q1 went 1.33x (W=0.15) -> 2.16x (W=0.30) against
# 3.44x uncapped -- the excess removed fell from ~87% to ~52% -- and
# rho(log A_FeO/A_OH, A_moon) +0.210 -> +0.396 against +0.484.  On gated rows
# the block moved -14.0% instead of -32.3%, FeO -27.9% instead of -47.7%.
# FeO amplitude MAD also worsened, 0.01791 -> 0.01948.
#
# WHAT IS STILL TRUE ABOUT THE 0.216 dex RISK NOTE: 0.15 dex is ~0.7x the
# dark-time robust sigma (one-sided p84-p50 = 0.305 dex), so the bound does sit
# inside the intrinsic dark-time spread and may clip legitimate variation on
# gated rows.  Measured, that costs little: blue chi2 on gated-but-not-binding
# rows -0.03%, dark rows -0.00%, collapse rate 2.28% -> 2.35%, decomposition
# reduced_chi2 median ratio 1.0000.  The one genuine argument for a looser
# bound is training stability -- seed-to-seed mean_eRMSE std 0.614 at W=0.15
# against 0.354 at 0.30 and 0.191 uncapped -- but mean_eRMSE is itself the
# absolute coefficient-space metric and inherits the same degeneracy, so that
# signal is not clean either.
#
# The CENTRE is unchanged and confirmed: c = -0.6489 against a measured
# dark-time median of -0.6616, agreeing to 0.013 dex.
#
# gaia-stars-mask-cont2 IS the corpus these settings produce: verified that
# cont2 and cont3 are 100.00% bit-identical on every ungated row in all three
# arms (852/852, 851/851, 844/844) and 0% identical on gated rows, and that
# sky_decomp/ was untouched between the two commits -- W was the only change.
#
# BASIS WARNING for anyone re-measuring this bound: the stored OH coefficients
# live on the convolved STICK basis, not on `matrix_oh`
# (sum(COMP_OH)/(coef . stick_rowsum) = 1.0005 against 0.19956 for
# matrix_oh.sum(axis=1)), while the diffuse coefficients DO live on
# matrix_diffuse.  A cross-family ratio built from matrix_* is 5.01x wrong on
# the OH side; integrate the stored COMP_* planes instead.  Measured that way
# the bound binds at log10 = -0.4924 against the specified -0.4989.
SPLIT_ZODI_DIFFUSE_OH_CENTRE_LOG10 = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_oh_centre_log10"
]
SPLIT_ZODI_DIFFUSE_OH_BOUND_DEX = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_oh_bound_dex"
]
SPLIT_ZODI_DIFFUSE_OH_GATE_FRAC = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_oh_gate_frac"
]
SPLIT_ZODI_DIFFUSE_OH_RELAX_DEX = SPLIT_ZODI_CONTINUUM_DEFAULTS[
    "diffuse_oh_relax_dex"
]
# Absolute recentring of the Leinert anchor.  The anchor brackets the fitted
# zodi total to [Z_pred/kappa_z, kappa_z * Z_pred], and Z_pred comes from
# _physics_only_model, whose learned scale factors are deliberately zeroed --
# so nothing has ever calibrated its ABSOLUTE normalisation.  Measured on the
# new-oh-2 ML test split (1113 rows) by comparing each fitted zodi total with
# that prediction, the anchor turned out to be SATURATED: 67.7% of all rows and
# 93.1% of moon-up rows sat exactly on the ceiling, every percentile p10-p99 of
# log10(Z_fit/Z_pred) equal to +0.3010 = log10(2.0) to four decimals.  The fit
# was not measuring zodi there, it was reporting kappa_z.
#
# The gap is multiplicative, not a pedestal: log10(Z_fit/Z_pred) has slope
# +0.08 (rho = +0.07) against log10(Z_pred), and a two-parameter c*Z + p fit
# does no better than pure c (23.2% vs 23.5% median error) with an unphysical
# NEGATIVE p.  So Zodi_bs is not absorbing a non-zodiacal continuum; the
# normalisation is simply low.
#
# The size must be measured where there is no moonlight to leak, and with the
# censoring undone -- the interior rows are interior BECAUSE |r| < log10(2), so
# their median is biased toward 1.  A censored-Gaussian MLE gives:
#     moon down  1.61x (sigma 0.274 dex)   <- leakage-free, this is the number
#     moon up    8.32x                     <- not calibration: a calibration
#                                             offset cannot depend on the moon
#                                             (per-quartile: 3.1/9.4/3.8/3.8x)
# The moon carrier is independently 1.35x low (uncensored, IQR 0.120 dex), so
# one shared physical_to_fit_flux_scale error of ~1.4x explains both families.
# That is why the correction is applied to the zodi TOTAL only: the moon
# FRACTION is a ratio through the same conversion, so a shared factor cancels
# and SPLIT_ZODI_AMP_PRIOR_TOL keeps policing moon-into-zodi leakage unchanged.
#
# Checked on the same 200 lunation-stratified sky spectra the bounds were
# adopted on.  Every guardrail holds -- reversals 2/167 (was 2/164), dark-time
# moon share 0.0234 (unchanged to four digits), median rms 0.987x -- and the
# improvement is concentrated where the anchor was worst: rows pinned to a
# bound fall 77% -> 64% overall and 86% -> 71% with the moon up.
#
# Choosing the WIDTH matters as much as the centre, because the bracket is
# [c/kappa_z, c*kappa_z] * Z_raw and Z_true ~ 1.6 * Z_raw, so kappa_z sets the
# bracket in physical units.  Swept at c = 1.6:
#   kappa_z  physical bracket   pinned all/up/dark   rms    rms(FLI>0.8)
#     1.25   [0.80, 1.25]         88% / 90% / 75%   1.0000     1.0000
#     2.00   [0.50, 2.00]         64% / 71% / 21%   0.9867     0.9805
#     3.00   [0.33, 3.00]         44% / 51% /  0%   0.9783     0.9558
# kappa_z = 2.0 is kept: it is ~1.1 sigma of the measured 0.274 dex dark-time
# spread, so it BOUNDS the zodi without dictating it.  1.25 reproduces today's
# fits almost exactly (rms 1.0000 in every lunation bin) because everything
# still sits on a barely-moved ceiling, and it makes 75% of dark-time targets
# synthetic; 3.0 frees dark time completely but gives bright-moon zodi 3x
# headroom against a QP that already demands 8.3x.
#
# Two things NOT to conclude from the surrounding diagnostics.  (a) Tightening
# SPLIT_ZODI_AMP_PRIOR_TOL does not substitute for this: at kappa_f 1.5 and 1.2
# reversals rose to 6 and 10 of 167, and at c = 1.0 tightening it changed
# nothing at all (zodi_tot x1.000) because the anchor already pins the rows it
# would act on.  (b) rho(zodi, B500) and rho(zodi, FLI) are NOT trustworthy
# while the anchor binds: on a bound, zodi_tot == kappa_z * c * Z_pred exactly,
# so those correlations partly measure the constraint.  On rows interior in
# both c = 1.0 and c = 1.6 the fitted zodi is identical to machine precision
# (1.000x, IQR 0.0000) and every config gives the same rho(B500) ~ 0.74,
# rho(FLI) ~ 0.08 and partial rho(FLI | B500) ~ -0.12.  Judge this constraint
# by pinning fraction, reversals and rms, not by those correlations.
SPLIT_ZODI_ZODI_PRIOR_CALIBRATION = 1.6
# Empirical correction to the Leinert zodi the anchor is built from -- see
# `ZODI_LEINERT_CORRECTIONS` in sky_decomp/moon_zodi_model.py for the fit and
# its validation.  The default, "lvm-ecl-2026-09", is the ecliptic-latitude/
# longitude correction (dark-time scatter 0.124 -> 0.102 dex, sci-arm zodi
# contrast slope 0.45/0.43 -> 0.74/0.94; on a 1000-row decomposition A/B it
# removes the dark-time fitted/anchor trend with |beta| and cuts dark anchor
# pinning 12% -> 1% at unchanged chi2).  "none" reproduces every decomposition
# made before 2026-09-24 bit for bit (LVMSKY_ZODI_CORRECTION=none).
#
# A SOURCE CONSTANT, deliberately, not a CLI flag: the anchor is evaluated in
# the worker processes, and a module global set at runtime in the parent does
# not reach spawned workers, whereas a constant is present in every import.
#
# The environment variable LVMSKY_ZODI_CORRECTION overrides it, for side-by-side
# A/B runs from one checkout.  That IS spawn-safe where a runtime global is not:
# the environment is inherited by every worker at spawn time, so parent and
# workers evaluate this line to the same value.
#
# Whatever this is set to is written to the primary header as ZODICORR and
# folded into the run fingerprint.  The moon-model cache reads ZODICORR and
# refuses to pair with a decomposition built under a different correction:
# the ML's zodi-ceiling rule assumes pinned zodi = S x zodi_po with ONE fitted
# S, which only holds if the cache's zodi_po and this anchor agree.  Changing
# it therefore means: re-decompose, rebuild the cache, retrain.
SPLIT_ZODI_ZODI_CORRECTION = os.environ.get("LVMSKY_ZODI_CORRECTION", "lvm-ecl-2026-09")
from skysub.sky_decomp.moon_zodi_model import ZODI_LEINERT_CORRECTIONS as _ZLC  # noqa: E402
if SPLIT_ZODI_ZODI_CORRECTION not in _ZLC:
    raise ValueError(f"SPLIT_ZODI_ZODI_CORRECTION={SPLIT_ZODI_ZODI_CORRECTION!r} is not one of "
                     f"{sorted(_ZLC)}")
# Moon_bs interior-knot count.  Deliberately NOT SkyDecomp.__init__'s default
# (25, with n_zodi_spline_knots 3): the deployed corpus and every measurement
# behind the SPLIT_ZODI_* bounds above use 11 moon / 1 zodi interior knots.
# The ratio bounds are per ADJACENT KNOT PAIR, so the same beta is looser the
# more knots there are -- changing these without re-validating the bounds
# changes how much colour freedom each family actually has.
MOON_N_KNOTS_DEFAULT = SPLIT_ZODI_CONTINUUM_DEFAULTS["n_spline_knots"]

# --- Science emission-line mask -------------------------------------------
# Nebular lines from the SCIENCE field are not sky, and NONE of them exists in
# the 388-component basis, so the QP is forced to absorb them into whatever it
# has.  Measured on every10 row 773 (expnum 39622, galactic b = +0.48, an
# inner-plane H II region): Halpha equivalent width 62.4 A in the science fibre
# against 10.0 and 2.8 in the two sky arms, and the FITTED OH component runs
# 6.7x its sideband level inside the Halpha/[NII] window and 2.0x inside
# [SII].  So the absorber is OH -- the only narrow-line family with lines
# there -- not the moon or zodi splines, which is why the science-continuum
# colour gate in mlp_predictor.data does not see these rows and why nebular
# equivalent width shows no corpus-wide correlation with moon or zodi
# distortion (rho -0.107 and +0.044).
#
# Zeroing IVAR is the whole mechanism: SkyDecompBase._fit_design selects on
# `np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)`, so masked pixels leave
# the QP, chi2, the dof count and R2 together, with no interpolation and no
# change to the wavelength grid.
#
# COST, measured against the EXACT deployed basis -- design matrix rebuilt
# from the stored per-row LSF surface and validated to 7e-16 against the
# stored COMP_OH.  A design matrix built from the INITIAL LSF instead is wrong
# by a factor of 12 on OH and must not be used for this; it also mislocates OH
# line peaks, which made [OIII]4959 look coincident with OH_314 when it is not.
#
# At the deployed widths (1.5 x FWHM, so +/-2.04 A at Halpha) the mask covers
# 85/12401 pixels = 0.69% and removes 0.96% of the OH model, 0.95% of the
# moon, 0.70% of the zodi, 0.76% of the diffuse block and 0.00% of ATOM.
# Three OH components lose more than 20% of their own support -- OH_316 64%,
# OH_144 45%, OH_341 32% -- carrying 0.238% of the OH flux between them
# (ranks 81-217 of 357), and each keeps support elsewhere: OH_341 at
# 4034-5475 A, OH_316 at 3964-8175, and OH_144 its 6555.5 A line, which the
# window leaves outside while masking its 6561.5 A one.
#
# Almost all of that cost is Halpha and [NII]6583.  [OII]3726/3729,
# [OIII]4959 and [OIII]5007 cost NOTHING -- no component loses even 10%.
# WIDTH IS THE SENSITIVE KNOB, and it is asymmetric: widening buys very little
# line flux (the core is already covered) and costs OH support quickly --
# 2.0 x FWHM takes OH from 0.96% to 1.09% and pulls OH_285 past 20%, and a
# +/-8 A window would take OH_144 to 94% (its whole support is two lines, at
# 6555.5 and 6561.5 A) and add OH_150.  Re-measure before widening.
#
# Both [NII] and both [SII] and [OIII]4959 are included even though only the
# brighter partner was asked for: they are fixed-ratio partners of lines
# already masked (1/3 of [NII]6583 and of [OIII]5007), they sit inside or
# beside the same windows, and leaving one of a doublet unmasked leaves the
# contaminant in the fit at a third of its strength.
SCIENCE_EMISSION_LINES = (
    ("[OII]3726", 3726.03),
    ("[OII]3729", 3728.82),
    ("Hbeta", 4861.33),
    ("[OIII]4959", 4958.91),
    ("[OIII]5007", 5006.84),
    ("[NII]6548", 6548.05),
    ("Halpha", 6562.80),
    ("[NII]6583", 6583.45),
    ("[SII]6716", 6716.44),
    ("[SII]6731", 6730.82),
)
# Half-width = max(MIN, FWHM_MULTIPLE * FWHM(lambda)) + lambda * v/c, scaled
# off the detector FWHM at each line rather than one assumed resolution.
#
# 1.5 x FWHM (3.53 sigma) masks the bright core, which is where essentially all
# the contaminating flux is, and deliberately leaves the wings in: doubling the
# window to 2.0 x FWHM buys almost no extra line flux while taking OH from
# 0.96% to 1.09% and pulling a fourth component past 20% support loss.
#
# A displaced line is not a wing effect -- the core itself moves -- so instead
# of widening the windows the mask SLIDES them, using a Halpha velocity
# measured per row (SCIENCE_LINE_MASK_CENTRE_ON_HALPHA).  The extra widening
# term SCIENCE_LINE_MASK_VELOCITY_KM_S is therefore 0 by default.
#
# THE CENTROID MUST BE MEASURED AGAINST A SKY REFERENCE.  There are OH lines
# directly under Halpha -- OH_144 at 6555.5/6561.5 A, OH_157 at 6559.5/6562.5,
# OH_285 at 6557.5/6561.5, OH_316 at 6563.0 -- and they are bright enough to
# drag the centroid: the two SKY fibres, which are essentially pure OH in this
# window, centroid at -3.08 and -4.20 A.  Measured on 2697 science fibres with
# Halpha EW excess > 10 A:
#
#   raw science flux      median -0.71 A (-32 km/s), p1 -4.33, p99 +4.12,
#                         87.9% within +/-2.04 A
#   minus a scaled sky arm  median +0.17 A ( +8 km/s), p1 -2.07, p99 +5.83,
#                         94.4% within +/-2.04 A
#
# So the true nebular velocities sit much closer to rest than the raw numbers
# suggest, and most of the apparent -32 km/s was OH.
#
# WHAT CENTRING ACTUALLY BUYS, validated on those 2697 rows: the pipeline
# measures a velocity on 92.5% of them (median +8.7 km/s over the whole
# every10 sample, matching the +8 km/s above; the other 73% of all rows have
# no detectable nebular Halpha and correctly fall back to rest).  Core
# coverage goes 89.0% -> 92.5%: of the 297 rows the rest-frame mask missed,
# centring covers 102, while 8 rows (0.30%) that WERE covered are now missed
# through a mis-measured velocity.  It is a tail fix worth roughly +3.5pp, not
# a bulk one -- do not expect it to change aggregate metrics.
#
# Masked pixel count stays 83-87 against the rest-frame 85, so dof moves by at
# most a couple of pixels and reduced_chi2 stays comparable across rows.
SCIENCE_LINE_MASK_ENABLED = True
SCIENCE_LINE_MASK_FWHM_MULTIPLE = 1.5
SCIENCE_LINE_MASK_VELOCITY_KM_S = 0.0
SCIENCE_LINE_MASK_MIN_HALF_WIDTH_A = 2.0
# Per-row centring: measure the nebular velocity from Halpha in the SCIENCE
# fibre and slide every window to the observed wavelength instead of widening
# it.  One velocity per row, applied to all ten lines and all three arms --
# the shift is a property of the emitting gas, not of the line, and keeping it
# common across arms means the three fits still exclude exactly the same
# pixels, which is what makes their coefficients comparable.
SCIENCE_LINE_MASK_CENTRE_ON_HALPHA = True
SCIENCE_LINE_MASK_MAX_SHIFT_KM_S = 300.0
SCIENCE_LINE_MASK_CENTRE_MIN_SNR = 5.0
_C_KM_S = 299792.458


def measure_halpha_velocity(
    wave,
    flux_sci,
    flux_sky=None,
    max_shift_km_s=SCIENCE_LINE_MASK_MAX_SHIFT_KM_S,
    min_snr=SCIENCE_LINE_MASK_CENTRE_MIN_SNR,
    search_half_width_a=12.0,
):
    """Nebular Halpha velocity of one row, in km/s, or 0.0 if not measurable.

    ``flux_sky`` is a sky-fibre spectrum used to cancel the OH lines that sit
    directly under Halpha; pass ``None`` only if none is available, and expect
    a blueward bias of a few km/s to tens of km/s if you do (the sky fibres
    centroid at -3.1 and -4.2 A in this window because of OH alone).  The sky
    is scaled by the ratio of positive flux in the window, capped at 1, so it
    can only remove the shared airglow, never add a negative pedestal deeper
    than the science spectrum itself.

    Returns 0.0 -- i.e. leave the windows at rest -- whenever the line is not
    convincingly detected, the centroid lands outside the search window, or
    the implied shift exceeds ``max_shift_km_s``.  Failing closed matters:
    a spurious shift moves the mask off a line that WAS being masked.
    """
    wave = np.asarray(wave, dtype=np.float64)
    lam0 = 6562.80
    lo, hi = lam0 - float(search_half_width_a), lam0 + float(search_half_width_a)
    i0, i1 = np.searchsorted(wave, [lo, hi])
    if int(i1) - int(i0) < 5:
        return 0.0
    sl = slice(int(i0), int(i1))
    w = wave[sl]
    y = np.asarray(flux_sci, dtype=np.float64)[sl]
    if flux_sky is not None:
        sky = np.asarray(flux_sky, dtype=np.float64)[sl]
        num = float(np.nansum(np.clip(y, 0.0, None)))
        den = float(np.nansum(np.clip(sky, 0.0, None)))
        if den > 0.0:
            y = y - min(num / den, 1.0) * sky
    # Local continuum from sidebands either side of the Halpha/[NII] complex.
    cw = (np.abs(wave - lam0) > 25.0) & (np.abs(wave - lam0) <= 70.0)
    base = np.asarray(flux_sci, dtype=np.float64)[cw]
    cont = float(np.nanmedian(base)) if np.any(np.isfinite(base)) else 0.0
    net = y - cont
    good = np.isfinite(net)
    if not np.any(good):
        return 0.0
    pos = np.where(good & (net > 0.0), net, 0.0)
    total = float(pos.sum())
    if total <= 0.0:
        return 0.0
    # Detection test against the scatter of the negative excursions, which is
    # what the window looks like when there is no line.
    neg = net[good & (net < 0.0)]
    noise = float(np.std(neg)) if neg.size >= 3 else 0.0
    peak = float(np.nanmax(pos))
    if noise > 0.0 and peak < float(min_snr) * noise:
        return 0.0
    centroid = float((pos * w).sum() / total)
    if not np.isfinite(centroid) or centroid <= lo or centroid >= hi:
        return 0.0
    v = (centroid - lam0) / lam0 * _C_KM_S
    if not np.isfinite(v) or abs(v) > float(max_shift_km_s):
        return 0.0
    return v


def science_line_mask(
    wave,
    lsf_fwhm=None,
    lines=SCIENCE_EMISSION_LINES,
    fwhm_multiple=SCIENCE_LINE_MASK_FWHM_MULTIPLE,
    velocity_km_s=SCIENCE_LINE_MASK_VELOCITY_KM_S,
    min_half_width_a=SCIENCE_LINE_MASK_MIN_HALF_WIDTH_A,
    centre_velocity_km_s=0.0,
):
    """Boolean mask over ``wave``, True where a science emission line sits.

    Each window is centred on ``lambda * (1 + centre_velocity_km_s/c)`` --
    normally the per-row Halpha velocity from ``measure_halpha_velocity``,
    applied to every line because the shift belongs to the emitting gas -- and
    has half-width
    ``max(min_half_width_a, fwhm_multiple * FWHM(lambda)) + lambda * v/c``.

    ``lsf_fwhm`` is the DETECTOR LSF FWHM in Angstrom -- the LSF_* arrays in
    the input FITS, which is what the pipeline actually carries (median 1.57 A
    on gaia1over100, 1.36 A at Halpha).  Do NOT pass ``--lsf-sigma`` here: that
    is a scalar Gaussian SIGMA defaulting to 0.5 A, a different quantity by a
    factor of 2.35.  A per-pixel array, a scalar, or ``None`` are all accepted;
    ``None`` (or an array with no usable pixel near a line) falls back to
    ``min_half_width_a``.

    Returns ``(mask, widths)``, ``widths`` being the per-line half-width in
    Angstrom so the caller can report what it actually masked.
    """
    wave = np.asarray(wave, dtype=np.float64)
    fwhm = None
    if lsf_fwhm is not None:
        arr = np.asarray(lsf_fwhm, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(wave.shape, float(arr))
        if arr.shape == wave.shape:
            fwhm = arr
    mask = np.zeros(wave.shape, dtype=bool)
    widths = {}
    shift = 1.0 + float(centre_velocity_km_s) / _C_KM_S
    for name, lam in lines:
        centre = float(lam) * shift
        half = float(min_half_width_a)
        if fwhm is not None:
            near = np.abs(wave - centre) <= 25.0
            usable = near & np.isfinite(fwhm) & (fwhm > 0)
            if np.any(usable):
                half = max(half,
                           float(fwhm_multiple) * float(np.median(fwhm[usable])))
        half += float(lam) * float(velocity_km_s) / _C_KM_S
        widths[name] = half
        mask |= (wave >= centre - half) & (wave <= centre + half)
    return mask, widths


def _science_mask_lsf_reference(wave, lsf_by_role):
    """Median detector FWHM only where the science-line mask needs it."""
    wave = np.asarray(wave, dtype=np.float64)
    reference = np.full(wave.shape, np.nan, dtype=np.float64)
    for _, wavelength in SCIENCE_EMISSION_LINES:
        use = np.abs(wave - float(wavelength)) <= 25.0
        arm_medians = []
        for values in lsf_by_role.values():
            values = np.asarray(values)
            arm_medians.append(
                values[use]
                if values.ndim == 1
                else np.nanmedian(values[:, use], axis=0)
            )
        if arm_medians:
            reference[use] = np.nanmedian(np.vstack(arm_medians), axis=0)
    return reference



def init_worker(
    wave,
    lsf_sigma,
    base_dir,
    factor,
    data_file,
    progress_queue=None,
    fit_model="baseline",
    n_refinement_cycles=5,
    worker_counter=None,
    pin_cpu=False,
    diagnose_threads=False,
    palace_suffix=None,
    palace_oh_suffix=None,
    palace_diffuse_suffix=None,
    exposure_seconds=900.0,
    moon_zodi_data_root=None,
    n_spline_knots=MOON_N_KNOTS_DEFAULT,
    n_zodi_spline_knots=SPLIT_ZODI_N_KNOTS_DEFAULT,
    zodi_smooth_lambda=SPLIT_ZODI_SMOOTH_LAMBDA_DEFAULT,
    moon_smooth_lambda=MOON_SMOOTH_LAMBDA_DEFAULT,
    mask_science_lines=SCIENCE_LINE_MASK_ENABLED,
    centre_on_halpha=SCIENCE_LINE_MASK_CENTRE_ON_HALPHA,
    fit_pixel_weights=FIT_PIXEL_WEIGHTS,
    fit_pixel_weight_clip=FIT_PIXEL_WEIGHT_CLIP,
    reversal_retry_bound=(
        SPLIT_ZODI_REVERSAL_RETRY_BOUND if SPLIT_ZODI_REVERSAL_RETRY else None
    ),
    diffuse_ratio_bound_dex=SPLIT_ZODI_DIFFUSE_RATIO_BOUND_DEX,
    diffuse_ratio_nominal=SPLIT_ZODI_DIFFUSE_RATIO_NOMINAL,
    diffuse_oh_centre_log10=SPLIT_ZODI_DIFFUSE_OH_CENTRE_LOG10,
    diffuse_oh_bound_dex=SPLIT_ZODI_DIFFUSE_OH_BOUND_DEX,
    compact_cache_dir=None,
    run_fingerprint=None,
):
    """Initialise one SkyDecomp instance per worker process."""
    global \
        _WORKER_DECOMPOSER, \
        _WORKER_FACTOR, \
        _WORKER_HDU, \
        _WORKER_FLUX, \
        _WORKER_LSF, \
        _WORKER_META, \
        _WORKER_PROGRESS_QUEUE, \
        _WORKER_FIT_MODEL, \
        _WORKER_EXPOSURE_SECONDS, \
        _WORKER_WAVE, \
        _WORKER_TELLURIC_CALCULATOR, \
        _WORKER_DECOMPOSER_KWARGS, \
        _WORKER_SCIENCE_LINE_MASK, \
        _WORKER_SCIENCE_LINE_FWHM, \
        _WORKER_SCIENCE_LINE_CENTRE, \
        _WORKER_FIT_PIXEL_WEIGHTS, \
        _WORKER_FIT_PIXEL_WEIGHT_CLIP, \
        _WORKER_LOG_WAVE, \
        _WORKER_REVERSAL_RETRY_BOUND, \
        _WORKER_SENS_ABS, \
        _WORKER_COMPACT_CACHE_DIR, \
        _WORKER_RUN_FINGERPRINT, \
        _PWV_FALLBACK_REPORTED

    _clamp_native_threads(1)
    palace_oh_suffix = _resolved_palace_oh_suffix(fit_model, palace_oh_suffix)

    worker_rank = 0
    if worker_counter is not None:
        with worker_counter.get_lock():
            worker_rank = int(worker_counter.value)
            worker_counter.value = worker_rank + 1
    if pin_cpu and hasattr(os, "sched_setaffinity"):
        try:
            available = sorted(os.sched_getaffinity(0))
            if available:
                target = available[worker_rank % len(available)]
                os.sched_setaffinity(0, {target})
        except OSError as exc:
            print(f"[worker pid={os.getpid()}] pin_cpu failed: {exc}", flush=True)

    _WORKER_FACTOR = float(factor)
    _WORKER_FIT_MODEL = fit_model
    _WORKER_EXPOSURE_SECONDS = float(exposure_seconds)
    _WORKER_WAVE = np.asarray(wave, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        _WORKER_LOG_WAVE = np.log(_WORKER_WAVE)
    _WORKER_TELLURIC_CALCULATOR = None
    _WORKER_DECOMPOSER_KWARGS = {}
    _WORKER_COMPACT_CACHE_DIR = compact_cache_dir
    _WORKER_RUN_FINGERPRINT = run_fingerprint
    _PWV_FALLBACK_REPORTED = False
    # Keep worker-local memmapped access to flux tables to avoid large IPC payloads.
    # An already-open HDUList (an in-memory stack, see `decompose_in_process`)
    # is used as is.
    _WORKER_HDU = (data_file if isinstance(data_file, fits.HDUList)
                   else fits.open(data_file, memmap=True))
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_FLUX = {
        "sci": np.asarray(_WORKER_HDU["FLUX_SCI"].data),
        "sky1": np.asarray(_WORKER_HDU["FLUX_SKY_NEAR"].data),
        "sky2": np.asarray(_WORKER_HDU["FLUX_SKY_FAR"].data),
    }
    # split-zodi needs the same LSF + META as the model-based mode: its
    # amplitude priors are geometry predictions, one per spectrum.
    if fit_model in (
        MOON_ZODI_FIT_MODEL,
        *SPLIT_ZODI_FIT_MODELS,
        *TELLURIC_FIT_MODELS,
    ):
        _WORKER_LSF = {
            "sci": np.asarray(_WORKER_HDU["LSF_SCI"].data),
            "sky1": np.asarray(_WORKER_HDU["LSF_SKY_NEAR"].data),
            "sky2": np.asarray(_WORKER_HDU["LSF_SKY_FAR"].data),
        }
        _WORKER_META = _WORKER_HDU["META"].data
    else:
        _WORKER_LSF = {}
        _WORKER_META = None
    if fit_model == "baseline":
        from skysub.sky_decomp.fit import SkyDecomp

        _WORKER_DECOMPOSER = SkyDecomp(
            wave,
            lsf_sigma=lsf_sigma,
            base_dir=base_dir,
            palace_suffix=palace_suffix,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            moon_smooth_lambda=0.1,
            moon_interline_boost=10000.0,
            moon_interline_red_min=6000.0,
            moon_interline_exclusion_a=2.5,
            moon_interline_line_flux_threshold=0.01,
        )
    elif fit_model == "lsf-surface-iterative":
        from skysub.sky_decomp.lsf_surface_iterative import (
            LSFSurfaceIterativeConfig,
            SkyDecompLSFSurfaceIterative,
        )

        _WORKER_DECOMPOSER = SkyDecompLSFSurfaceIterative(
            wave,
            lsf_sigma=lsf_sigma,
            base_dir=base_dir,
            palace_suffix=palace_suffix,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            moon_smooth_lambda=0.1,
            moon_interline_boost=0.0,
            n_spline_knots=int(n_spline_knots),
            config=LSFSurfaceIterativeConfig(
                n_refinement_cycles=n_refinement_cycles,
            ),
        )
    elif fit_model in SPLIT_ZODI_FIT_MODELS:
        from skysub.sky_decomp.lsf_surface_iterative import (
            LSFSurfaceIterativeConfig,
            SkyDecompLSFSurfaceIterative,
        )

        decomposer_class = SkyDecompLSFSurfaceIterative
        if fit_model == SPLINE2D_SPLIT_ZODI_FIT_MODEL:
            from skysub.sky_decomp.lsf_spline2d import SkyDecompLSFSpline2D

            decomposer_class = SkyDecompLSFSpline2D
        _WORKER_DECOMPOSER = decomposer_class(
            wave,
            lsf_sigma=lsf_sigma,
            base_dir=base_dir,
            palace_suffix=palace_suffix,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            moon_smooth_lambda=0.1,
            moon_interline_boost=0.0,
            n_spline_knots=int(n_spline_knots),
            split_zodi=True,
            n_zodi_spline_knots=int(n_zodi_spline_knots),
            zodi_smooth_lambda=float(zodi_smooth_lambda),
            moon_albedo_fiducial_phase_deg=SPLIT_ZODI_MOON_ALBEDO_PHASE_DEG,
            zodi_color_exponent=SPLIT_ZODI_COLOR_EXPONENT,
            moon_ratio_bound=SPLIT_ZODI_MOON_RATIO_BOUND,
            zodi_ratio_bound=SPLIT_ZODI_ZODI_RATIO_BOUND,
            amp_prior_tol=SPLIT_ZODI_AMP_PRIOR_TOL,
            zodi_amp_bound=SPLIT_ZODI_ZODI_AMP_BOUND,
            diffuse_ratio_bound_dex=float(diffuse_ratio_bound_dex),
            diffuse_ratio_nominal=diffuse_ratio_nominal,
            diffuse_oh_centre_log10=diffuse_oh_centre_log10,
            diffuse_oh_bound_dex=float(diffuse_oh_bound_dex),
            diffuse_oh_gate_frac=SPLIT_ZODI_DIFFUSE_OH_GATE_FRAC,
            diffuse_oh_relax_dex=SPLIT_ZODI_DIFFUSE_OH_RELAX_DEX,
            diffuse_oh_scope="block",
            config=LSFSurfaceIterativeConfig(
                n_refinement_cycles=n_refinement_cycles,
                **(
                    {"roughness_fraction": 1.0e-4}
                    if fit_model == SPLINE2D_SPLIT_ZODI_FIT_MODEL
                    else {}
                ),
            ),
        )
    elif fit_model == MOON_ZODI_FIT_MODEL:
        from skysub.sky_decomp.lsf_surface_iterative import LSFSurfaceIterativeConfig
        from skysub.sky_decomp.moon_zodi_lsf_surface_iterative import (
            SkyDecompMoonZodiLSFSurfaceIterative,
        )
        _WORKER_DECOMPOSER = SkyDecompMoonZodiLSFSurfaceIterative(
            wave,
            lsf_sigma=lsf_sigma,
            data_root=(
                DEFAULT_MOON_ZODI_DATA_ROOT
                if moon_zodi_data_root is None
                else moon_zodi_data_root
            ),
            palace_suffix=palace_suffix,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            moon_smooth_lambda=0.1,
            moon_interline_boost=0.0,
            physical_to_fit_flux_scale=float(factor),
            config=LSFSurfaceIterativeConfig(
                n_refinement_cycles=n_refinement_cycles,
            ),
        )
    elif fit_model in TELLURIC_FIT_MODELS:
        from lvmdrp.core.fluxcal import TelluricCalculator
        from skysub.sky_decomp.lsf_surface_iterative import LSFSurfaceIterativeConfig

        if fit_model == ADAM25K_TELLURIC_FIT_MODEL:
            from skysub.sky_decomp.telluric_corrected_lines import (
                SkyDecompAdam25kTelluricLSFSpline2D,
            )

            _WORKER_DECOMPOSER = SkyDecompAdam25kTelluricLSFSpline2D
        elif fit_model == PALACE_VNF_PCA30_FIT_MODEL:
            from skysub.sky_decomp.residual_pca import (
                SkyDecompPalaceAijcVNFLineAmplitudePCA,
            )

            _WORKER_DECOMPOSER = SkyDecompPalaceAijcVNFLineAmplitudePCA
        elif fit_model in ADAM25K_SPLIT_ZODI_FIT_MODELS:
            from skysub.sky_decomp.telluric_corrected_lines import (
                SkyDecompAdam25kTelluricSplitZodiLSFSpline2D,
            )

            _WORKER_DECOMPOSER = SkyDecompAdam25kTelluricSplitZodiLSFSpline2D
        elif fit_model == PALACE_VNF_SPLIT_ZODI_FIT_MODEL:
            from skysub.sky_decomp.residual_pca import (
                SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D,
            )

            _WORKER_DECOMPOSER = SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D
        elif fit_model == PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL:
            from skysub.sky_decomp.residual_pca import (
                SkyDecompPalaceCorrAijcVNFSplitZodiLSFSpline2D,
            )

            _WORKER_DECOMPOSER = SkyDecompPalaceCorrAijcVNFSplitZodiLSFSpline2D
        else:
            from skysub.sky_decomp.residual_pca import (
                SkyDecompPalaceAijcVNFSplitZodiLineAmplitudePCA30,
            )

            _WORKER_DECOMPOSER = SkyDecompPalaceAijcVNFSplitZodiLineAmplitudePCA30
        _WORKER_TELLURIC_CALCULATOR = TelluricCalculator()
        _WORKER_DECOMPOSER_KWARGS = {
            "lsf_sigma": lsf_sigma,
            "base_dir": base_dir,
            "palace_suffix": palace_suffix,
            "palace_oh_suffix": palace_oh_suffix,
            "palace_diffuse_suffix": palace_diffuse_suffix,
            "moon_smooth_lambda": float(moon_smooth_lambda),
            "moon_interline_boost": 0.0,
            "n_spline_knots": int(n_spline_knots),
            "config": LSFSurfaceIterativeConfig(
                n_refinement_cycles=n_refinement_cycles,
                roughness_fraction=1.0e-4,
            ),
        }
        if fit_model in SPLIT_ZODI_TELLURIC_FIT_MODELS:
            _WORKER_DECOMPOSER_KWARGS.update(
                n_zodi_spline_knots=int(n_zodi_spline_knots),
                zodi_smooth_lambda=float(zodi_smooth_lambda),
                diffuse_ratio_bound_dex=float(diffuse_ratio_bound_dex),
                diffuse_ratio_nominal=diffuse_ratio_nominal,
                diffuse_oh_centre_log10=diffuse_oh_centre_log10,
                diffuse_oh_bound_dex=float(diffuse_oh_bound_dex),
            )
        if fit_model in (
            PALACE_VNF_PCA30_FIT_MODEL,
            *PALACE_VNF_PCA30_SPLIT_ZODI_FIT_MODELS,
        ):
            _WORKER_DECOMPOSER_KWARGS["n_line_amplitude_pca_components"] = 30
    else:
        raise ValueError(f"Unknown fit model: {fit_model}")

    # Science emission-line mask, built once per worker.  Deliberately built
    # from the init-time LSF rather than per row: the row-to-row LSF variation
    # is small next to the +/-150 km/s velocity term, and a row-dependent mask
    # would make the number of fitted pixels vary from row to row, which
    # reduced_chi2 and the dof count would then carry.
    _WORKER_SCIENCE_LINE_MASK = None
    _WORKER_SCIENCE_LINE_FWHM = None
    _WORKER_SCIENCE_LINE_CENTRE = bool(centre_on_halpha)
    _WORKER_REVERSAL_RETRY_BOUND = (
        None if reversal_retry_bound is None else float(reversal_retry_bound)
    )
    _WORKER_FIT_PIXEL_WEIGHTS = bool(fit_pixel_weights)
    _WORKER_FIT_PIXEL_WEIGHT_CLIP = (
        None if fit_pixel_weight_clip is None else float(fit_pixel_weight_clip)
    )
    _WORKER_SENS_ABS = None
    if _WORKER_FIT_PIXEL_WEIGHTS:
        from skysub.sky_decomp.pixel_weights import absolute_sensitivity
        try:
            _WORKER_SENS_ABS = absolute_sensitivity(
                wave, verbose=(worker_rank == 0))
        except RuntimeError as error:
            # Deliberately fatal rather than a silent fallback: the weights are
            # ON by default now, and quietly producing an UNWEIGHTED corpus that
            # everyone believes is weighted is the worse failure by far.  Say
            # what to do about it.
            raise RuntimeError(
                f"{error}  The per-pixel photon weights are on by default and "
                f"need the absolute sensitivity across the whole fitted grid. "
                f"Pass --no-fit-pixel-weights to fit unweighted instead (the "
                f"pre-2026-09-18 behaviour), or restrict the wavelength grid to "
                f"the covered range.") from error
        if worker_rank == 0:
            print('[fit weights] per-pixel inverse-variance weighting ON '
                  '(row-normalised to mean 1; reduced_chi2 is now a '
                  'WEIGHTED chi2 and not comparable to earlier runs)')
    if mask_science_lines:
        # Use the DETECTOR LSF FWHM from the input FITS, median-combined over
        # rows and arms, not the scalar `lsf_sigma` argument -- that is a
        # 0.5 A Gaussian sigma by default while the real FWHM is ~1.57 A, so
        # feeding it here would size every window off the wrong quantity.
        # _WORKER_LSF is empty for fit models that do not need it; then the
        # instrumental term is simply dropped.
        _fwhm_ref = (
            _science_mask_lsf_reference(wave, _WORKER_LSF)
            if _WORKER_LSF
            else None
        )
        # Use the `wave` ARGUMENT, not _WORKER_DECOMPOSER.wave: every
        # decomposer is constructed from this same array, and reading it off
        # the object couples worker init to the concrete decomposer class
        # (test_palace_suffix's FakeDecomposer has no `.wave`).
        _mask, _widths = science_line_mask(wave, _fwhm_ref)
        _WORKER_SCIENCE_LINE_MASK = _mask
        # Kept so fit_chunk_worker can rebuild the mask per row at the
        # measured Halpha velocity; the static mask above stays as the
        # fallback for rows where the line is not measurable.
        _WORKER_SCIENCE_LINE_FWHM = _fwhm_ref
        if worker_rank == 0:
            print(f"[science-line mask] {int(_mask.sum())}/{_mask.size} pixels "
                  f"({100.0 * _mask.mean():.2f}%) excluded via IVAR=0 in "
                  f"{len(SCIENCE_EMISSION_LINES)} windows: "
                  + ", ".join(f"{n} +/-{w:.1f}A" for n, w in _widths.items())
                  + ("; centred per row on the measured Halpha velocity"
                     if _WORKER_SCIENCE_LINE_CENTRE else "; fixed at rest"),
                  flush=True)

    # After all heavy imports, clamp once more and (optionally) report per-worker state.
    _clamp_native_threads(1)
    if diagnose_threads and worker_rank == 0:
        _report_thread_diagnostics()


def _report_thread_diagnostics():
    lines = [f"[worker pid={os.getpid()}] thread diagnostics:"]
    lines.append(
        f"  affinity_cores={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'n/a'}"
    )
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        lines.append(f"  {var}={os.environ.get(var, 'unset')}")
    if threadpool_limits is not None:
        try:
            from threadpoolctl import threadpool_info

            for entry in threadpool_info():
                lines.append(
                    f"  loaded_pool: {entry.get('user_api', '?'):8s} "
                    f"{entry.get('prefix', '?'):18s} threads={entry.get('num_threads', '?')}"
                )
        except Exception as exc:
            lines.append(f"  threadpool_info failed: {exc}")
    else:
        lines.append("  (threadpoolctl not installed; cannot enumerate loaded pools)")
    print("\n".join(lines), flush=True)


def _text_value(value):
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()


def _moon_zodi_observation(kind, row_index):
    from skysub.sky_decomp.moon_zodi_model import MoonZodiObservation

    role_contract = {
        "sci": ("sci", "sci_ra", "sci_dec"),
        "sky1": ("sky_near", "sky_near_ra", "sky_near_dec"),
        "sky2": ("sky_far", "sky_far_ra", "sky_far_dec"),
    }
    role, ra_column, dec_column = role_contract[kind]
    row = _WORKER_META[row_index]
    names = set(_WORKER_META.dtype.names or ())
    exposure = None
    for column in ("exposure_seconds", "exptime"):
        if column in names:
            candidate = float(row[column])
            if np.isfinite(candidate) and candidate > 0.0:
                exposure = candidate
                break
    if exposure is None:
        exposure = _WORKER_EXPOSURE_SECONDS
        exposure_source = "assumed_900s"
    else:
        exposure_source = "metadata"
    return MoonZodiObservation(
        expnum=int(row["expnum"]),
        date_obs=_text_value(row["date_obs"]),
        role=role,
        target_ra_deg=float(row[ra_column]),
        target_dec_deg=float(row[dec_column]),
        exposure_seconds=float(exposure),
        exposure_seconds_source=exposure_source,
    )


def _telluric_decomposer(kind, row_index, *, schema_only=False):
    global _PWV_FALLBACK_REPORTED

    from skysub.sky_decomp.telluric_corrected_lines import calculate_drp_transmission

    row = _WORKER_META[row_index]
    names = set(_WORKER_META.dtype.names or ())
    required = {
        "sci_airmass",
        "skye_airmass",
        "skyw_airmass",
        "sky_near_label",
        "sky_far_label",
    }
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"Telluric fit requires META columns: {', '.join(missing)}")

    raw_pwv = float(row["pwv_med"]) if "pwv_med" in names else np.nan
    pwv_mm = raw_pwv
    if not np.isfinite(pwv_mm) or pwv_mm <= 0.0:
        pwv_mm = DRP_DEFAULT_PWV_MM
        if not _PWV_FALLBACK_REPORTED:
            warnings.warn(
                f"Invalid or missing META.pwv_med ({raw_pwv!r}) at row {row_index}; "
                f"using the LVM DRP default PWV={DRP_DEFAULT_PWV_MM} mm",
                RuntimeWarning,
                stacklevel=2,
            )
            _PWV_FALLBACK_REPORTED = True
    sci_airmass = float(row["sci_airmass"])
    if kind == "sci":
        source_airmass = sci_airmass
    else:
        label_column = "sky_near_label" if kind == "sky1" else "sky_far_label"
        label = _text_value(row[label_column]).lower()
        airmass_column = {"skye": "skye_airmass", "skyw": "skyw_airmass"}.get(label)
        if airmass_column is None:
            raise ValueError(f"Unknown {label_column} value: {label!r}")
        source_airmass = float(row[airmass_column])
    invalid_airmass = not all(
        np.isfinite(value) and value > 0.0
        for value in (sci_airmass, source_airmass)
    )
    if invalid_airmass and not schema_only:
        raise _InvalidAirmassError(
            "invalid_airmass: "
            f"row={row_index}, role={kind}, sci_airmass={sci_airmass!r}, "
            f"source_airmass={source_airmass!r}"
        )
    if invalid_airmass:
        # Construct only the result schema; every fitted value returned below
        # is replaced by NaN, so this neutral airmass is never scientific data.
        sci_airmass = source_airmass = 1.0

    lsf_row = _sanitised_lsf_row(kind, row_index)
    if lsf_row is None and not schema_only:
        reason = _unusable_lsf_reason(kind, row_index)
        _report_unusable_lsf(kind, row_index, reason)
        raise _UnusableLSFError(reason)
    if lsf_row is None:
        # Construct only the result schema; every fitted value returned below
        # is replaced by NaN, so this placeholder LSF -- a flat 1 A FWHM, the
        # right order for LVM -- is never scientific data.
        lsf_row = np.ones_like(_WORKER_WAVE, dtype=np.float64)
    drp_transmission = calculate_drp_transmission(
        _WORKER_WAVE,
        lsf_row[None, :],
        pwv_mm,
        sci_airmass,
        _WORKER_TELLURIC_CALCULATOR,
    )
    return _WORKER_DECOMPOSER(
        _WORKER_WAVE,
        telluric_calculator=_WORKER_TELLURIC_CALCULATOR,
        pwv_mm=pwv_mm,
        source_airmass=source_airmass,
        drp_transmission=drp_transmission,
        **_WORKER_DECOMPOSER_KWARGS,
    )


_LSF_REPAIR_COUNT = {"rows": 0, "pixels": 0, "reported": False}
_LSF_UNUSABLE_REPORTED = False


def _lsf_row_defect(lsf):
    """Classify one detector LSF row.

    Returns the finite-positive mask and, when the row cannot be repaired, a
    short reason.  `_sanitised_lsf_row` and `_unusable_lsf_reason` both go
    through this so the rule that refuses a row and the reason recorded on
    that row's failed result can never disagree.
    """
    good = np.isfinite(lsf) & (lsf > 0.0)
    if good.all():
        return good, None
    if not good.any():
        return good, "no finite positive pixel"
    bad = np.flatnonzero(~good)
    if np.any(bad == 0) or np.any(bad == lsf.size - 1):
        return good, "a bad pixel at the array edge"
    if np.any(~good[bad - 1]) or np.any(~good[bad + 1]):
        return good, "adjacent bad pixels"
    return good, None


def _unusable_lsf_reason(kind, row_index):
    """Reason string recorded on a row whose detector LSF cannot be used."""
    lsf = np.asarray(_WORKER_LSF[kind][row_index], dtype=np.float64)
    good, reason = _lsf_row_defect(lsf)
    return (
        f"unusable_lsf: row={row_index}, role={kind}, "
        f"finite_positive_pixels={int(good.sum())}/{good.size}, "
        f"{reason or 'usable'}"
    )


def _report_unusable_lsf(kind, row_index, reason):
    """Warn once per worker; every such row carries its reason in the output."""
    global _LSF_UNUSABLE_REPORTED
    if _LSF_UNUSABLE_REPORTED:
        return
    _LSF_UNUSABLE_REPORTED = True
    warnings.warn(
        f"{reason}; this row is written as a failed_input NaN row.  Further "
        f"unusable-LSF rows in this worker are silent -- each one carries "
        f"fit_status=failed_input and its own reason in the output.",
        RuntimeWarning,
        stacklevel=2,
    )


def _sanitised_lsf_row(kind, row_index):
    """Return this row's detector LSF FWHM after isolated-pixel repair.

    The LSF is produced upstream and is normally clean: new-oh-3 has zero bad
    pixels in 17260 rows x 3 arms.  The gaia1over100 selection exposed a
    different set of fibres, 9 of which carry a single LSF pixel of exactly 0.0
    at a spectrograph arm join -- 8 at 5800.0 A (b/r) and 1 at 7570.0 A (r/z),
    10 bad pixels in 179 million.  ``MoonZodiPhysicalModel.predict`` rightly
    rejects a non-positive FWHM, which killed the whole worker chunk.

    The repaired row is used only for the DRP transmission and geometry prior;
    the decomposition still recovers its continuous LSF surface from the
    spectrum. A bad edge pixel or adjacent bad pixels are not repaired:
    returning ``None`` makes the caller fail or skip the prior rather than fit
    against a fabricated LSF curve.
    """
    lsf = np.asarray(_WORKER_LSF[kind][row_index], dtype=np.float64)
    good, reason = _lsf_row_defect(lsf)
    if reason is not None:
        return None
    if good.all():
        return lsf
    idx = np.arange(lsf.size)
    repaired = lsf.copy()
    repaired[~good] = np.interp(idx[~good], idx[good], lsf[good])
    _LSF_REPAIR_COUNT["rows"] += 1
    _LSF_REPAIR_COUNT["pixels"] += int((~good).sum())
    if not _LSF_REPAIR_COUNT["reported"]:
        _LSF_REPAIR_COUNT["reported"] = True
        _bad_w = _WORKER_WAVE[~good]
        print(
            f"  [lsf-repair] {kind} row {row_index}: interpolated "
            f"{int((~good).sum())} non-positive/non-finite LSF pixel(s) at "
            f"{', '.join(f'{x:.1f}' for x in _bad_w[:4])} A"
            f"{' ...' if _bad_w.size > 4 else ''}.  Further repairs in this "
            f"worker are silent; the total is reported at the end.",
            flush=True,
        )
    return repaired


_COMPACT_META_FIELDS = (
    "t_o2",
    "t_o2_err",
    "o2_prefit_amp",
    "reduced_chi2",
    "r2",
    "rms_resid",
    "resid_level",
    "fit_status",
    "fit_summary",
    "fit_elapsed_sec",
    "peak_memory_mb",
    "o2_fit_status",
    "o2_fit_summary",
    "o2_fit_elapsed_sec",
    "o2_valid_frac",
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _compact_cache_path(kind, row_index):
    return Path(_WORKER_COMPACT_CACHE_DIR) / kind / f"row-{row_index:05d}.npz"


def _compact_row_fingerprint(kind, row_index):
    return hashlib.sha256(
        f"{_WORKER_RUN_FINGERPRINT}:{kind}:{row_index}".encode()
    ).hexdigest()


def _compact_cache_is_current(kind, row_index):
    path = _compact_cache_path(kind, row_index)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["fingerprint"].item()) == _compact_row_fingerprint(
                kind, row_index
            )
    except (OSError, ValueError, KeyError):
        return False


def _compact_result_payload(kind, row_index, result):
    state = result.lsf_state
    state_meta = {
        "degrees": state.degrees,
        "channel_bounds": state.channel_bounds,
        "config": state.config,
        "metrics": state.metrics,
        "requested_cycles": state.requested_cycles,
        "completed_cycles": state.completed_cycles,
        "wave_n": state.wave_n,
        "wave_min": state.wave_min,
        "wave_max": state.wave_max,
        "wave_sha256": state.wave_sha256,
        "fit_status": state.fit_status,
        "failure_reason": state.failure_reason,
        "final_continuum_status": state.final_continuum_status,
        "final_line_status": state.final_line_status,
        "knot_strategy": state.knot_strategy,
        "legacy_kernel_representation": state.legacy_kernel_representation,
        "schema_version": state.schema_version,
    }
    payload = {
        "fingerprint": np.asarray(_compact_row_fingerprint(kind, row_index)),
        "success": np.asarray(True),
        "kind": np.asarray(kind),
        "source_row": np.asarray(row_index, dtype=np.int64),
        "meta_json": np.asarray(
            json.dumps(
                {name: getattr(result, name) for name in _COMPACT_META_FIELDS},
                sort_keys=True,
                default=_json_default,
            )
        ),
        "design_names": np.asarray(result.design_names),
        "coef": np.asarray(result.coef, dtype=np.float64),
        "coef_err": np.asarray(result.coef_err, dtype=np.float64),
        "lsf_state_json": np.asarray(
            json.dumps(state_meta, sort_keys=True, default=_json_default)
        ),
        "lsf_tap_offsets": np.asarray(state.tap_offsets, dtype=np.int64),
    }
    for channel in ("B", "R", "Z"):
        payload[f"lsf_coeff_{channel}"] = np.asarray(
            state.coefficients[channel], dtype=np.float64
        )
        payload[f"lsf_knots_{channel}"] = np.asarray(
            state.knot_vectors[channel], dtype=np.float64
        )
    for name in ("coef_cov_moon", "coef_cov_zodi"):
        value = getattr(result, name, None)
        payload[name] = (
            np.empty((0, 0), dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )
    return payload


def _save_compact_cache(kind, row_index, result=None, error=None, flags=None):
    path = _compact_cache_path(kind, row_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    if error is None:
        payload = _compact_result_payload(kind, row_index, result)
        if flags is not None:
            payload["reliability_json"] = np.asarray(
                json.dumps({k: (int(v) if k == "reliability" else float(v))
                            for k, v in flags.items()}, sort_keys=True))
    else:
        payload = {
            "fingerprint": np.asarray(_compact_row_fingerprint(kind, row_index)),
            "success": np.asarray(False),
            "kind": np.asarray(kind),
            "source_row": np.asarray(row_index, dtype=np.int64),
            "meta_json": np.asarray(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    sort_keys=True,
                )
            ),
        }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _load_cached_lsf_state(data):
    from skysub.sky_decomp.lsf_surface_iterative import LSFSurfaceState

    meta = json.loads(str(data["lsf_state_json"].item()))
    return LSFSurfaceState(
        coefficients={
            channel: np.asarray(data[f"lsf_coeff_{channel}"], dtype=np.float64)
            for channel in ("B", "R", "Z")
        },
        knot_vectors={
            channel: np.asarray(data[f"lsf_knots_{channel}"], dtype=np.float64)
            for channel in ("B", "R", "Z")
        },
        degrees={key: int(value) for key, value in meta["degrees"].items()},
        channel_bounds={
            key: tuple(value) for key, value in meta["channel_bounds"].items()
        },
        tap_offsets=np.asarray(data["lsf_tap_offsets"], dtype=np.int64),
        config=meta["config"],
        metrics=meta["metrics"],
        requested_cycles=int(meta["requested_cycles"]),
        completed_cycles=int(meta["completed_cycles"]),
        wave_n=int(meta["wave_n"]),
        wave_min=float(meta["wave_min"]),
        wave_max=float(meta["wave_max"]),
        wave_sha256=str(meta["wave_sha256"]),
        fit_status=str(meta["fit_status"]),
        failure_reason=str(meta["failure_reason"]),
        final_continuum_status=str(meta["final_continuum_status"]),
        final_line_status=str(meta["final_line_status"]),
        knot_strategy=str(meta["knot_strategy"]),
        legacy_kernel_representation=str(meta["legacy_kernel_representation"]),
        schema_version=int(meta["schema_version"]),
    )


def _write_compact_fits(
    cache_root, kind, n_rows, run_fingerprint, fit_model, output, primary_meta=None
):
    """Assemble one coefficient-only FITS from resumable per-row caches."""
    from astropy.table import Table
    from skysub.sky_decomp.result_io import _lsf_meta_row

    paths = [Path(cache_root) / kind / f"row-{index:05d}.npz" for index in range(n_rows)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} compact cache rows; first missing: {missing[0]}"
        )

    reference_names = reference_state = None
    moon_cov_shape = zodi_cov_shape = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if bool(data["success"].item()):
                reference_names = np.asarray(data["design_names"]).astype(str)
                reference_state = _load_cached_lsf_state(data)
                moon = np.asarray(data["coef_cov_moon"])
                zodi = np.asarray(data["coef_cov_zodi"])
                moon_cov_shape = moon.shape if moon.size else None
                zodi_cov_shape = zodi.shape if zodi.size else None
                break
    if reference_names is None or reference_state is None:
        raise RuntimeError(f"No successful {kind} fit is available for FITS assembly")

    n_coef = int(reference_names.size)
    coefficients = np.full((n_rows, n_coef), np.nan, dtype=np.float64)
    coefficient_errors = np.full_like(coefficients, np.nan)
    max_basis = max(
        reference_state.coefficients[channel].shape[1]
        for channel in ("B", "R", "Z")
    )
    max_knots = max(
        reference_state.knot_vectors[channel].size for channel in ("B", "R", "Z")
    )
    lsf_coefficients = np.full(
        (n_rows, 3, reference_state.tap_offsets.size, max_basis),
        np.nan,
        dtype=np.float64,
    )
    lsf_knots = np.full((n_rows, 3, max_knots), np.nan, dtype=np.float64)
    moon_cov = (
        None
        if moon_cov_shape is None
        else np.full((n_rows, *moon_cov_shape), np.nan, dtype=np.float64)
    )
    zodi_cov = (
        None
        if zodi_cov_shape is None
        else np.full((n_rows, *zodi_cov_shape), np.nan, dtype=np.float64)
    )
    meta_rows = []
    lsf_meta_rows = []
    success_count = 0
    for row_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            expected = hashlib.sha256(
                f"{run_fingerprint}:{kind}:{row_index}".encode()
            ).hexdigest()
            if str(data["fingerprint"].item()) != expected:
                raise ValueError(f"Stale compact cache: {path}")
            success = bool(data["success"].item())
            stored_meta = json.loads(str(data["meta_json"].item()))
            if success:
                names = np.asarray(data["design_names"]).astype(str)
                if not np.array_equal(names, reference_names):
                    raise ValueError(f"Coefficient schema changed at {path}")
                coefficients[row_index] = np.asarray(data["coef"], dtype=np.float64)
                coefficient_errors[row_index] = np.asarray(
                    data["coef_err"], dtype=np.float64
                )
                state = _load_cached_lsf_state(data)
                for channel_index, channel in enumerate(("B", "R", "Z")):
                    value = state.coefficients[channel]
                    knots = state.knot_vectors[channel]
                    lsf_coefficients[
                        row_index, channel_index, :, : value.shape[1]
                    ] = value
                    lsf_knots[row_index, channel_index, : knots.size] = knots
                    lsf_meta_rows.append(_lsf_meta_row(row_index, state, channel))
                if moon_cov is not None and np.asarray(data["coef_cov_moon"]).size:
                    moon_cov[row_index] = np.asarray(data["coef_cov_moon"])
                if zodi_cov is not None and np.asarray(data["coef_cov_zodi"]).size:
                    zodi_cov[row_index] = np.asarray(data["coef_cov_zodi"])
                success_count += 1
                error_type = error_message = ""
            else:
                for channel in ("B", "R", "Z"):
                    lsf_meta_rows.append(
                        _lsf_meta_row(
                            row_index,
                            reference_state,
                            channel,
                            available=False,
                            reason=stored_meta["error_message"],
                        )
                    )
                error_type = stored_meta["error_type"]
                error_message = stored_meta["error_message"]
                stored_meta = {name: np.nan for name in _COMPACT_META_FIELDS}
                for name in ("fit_status", "o2_fit_status"):
                    stored_meta[name] = "failed_input"
                for name in ("fit_summary", "o2_fit_summary"):
                    stored_meta[name] = ""
                stored_meta["fit_summary"] = error_message
            meta_rows.append(
                {
                    "source_row": row_index,
                    "role": kind,
                    "input_valid": success,
                    "error_type": error_type,
                    "error_message": error_message,
                    **stored_meta,
                }
            )

    primary = fits.PrimaryHDU()
    primary.header["DECOMPM"] = fit_model
    primary.header["COMPACT"] = True
    primary.header["RUNFP"] = run_fingerprint
    primary.header["NINPUT"] = n_rows
    primary.header["NSUCC"] = success_count
    for key, value in (primary_meta or {}).items():
        primary.header[key] = value
    coefficient_table = Table(
        {name: coefficients[:, index] for index, name in enumerate(reference_names)}
    )
    coefficient_error_table = Table(
        {
            name: coefficient_errors[:, index]
            for index, name in enumerate(reference_names)
        }
    )
    coefficient_hdu = fits.ImageHDU(lsf_coefficients, name="LSF_COEF")
    coefficient_hdu.header["TAPMIN"] = int(reference_state.tap_offsets[0])
    coefficient_hdu.header["TAPMAX"] = int(reference_state.tap_offsets[-1])
    coefficient_hdu.header["BASIS"] = "M-spline"
    hdus = [
        primary,
        fits.BinTableHDU(Table(rows=meta_rows), name="META"),
        fits.BinTableHDU(coefficient_table, name="COEF"),
        fits.BinTableHDU(coefficient_error_table, name="COEF_ERR"),
        coefficient_hdu,
        fits.ImageHDU(lsf_knots, name="LSF_KNOTS"),
        fits.BinTableHDU(Table(rows=lsf_meta_rows), name="LSF_META"),
    ]
    if moon_cov is not None and moon_cov_shape[0] > 1:
        hdus.append(fits.ImageHDU(moon_cov, name="COEF_COV_MOON"))
    if zodi_cov is not None and zodi_cov_shape[0] > 1:
        hdus.append(fits.ImageHDU(zodi_cov, name="COEF_COV_ZODI"))
    output = Path(output)
    temporary = output.with_suffix(".tmp.fits")
    fits.HDUList(hdus).writeto(temporary, overwrite=True)
    os.replace(temporary, output)
    print(
        f"Wrote compact {kind}: {success_count}/{n_rows} successful, "
        f"{n_coef} coefficients -> {output}"
    )


def _install_split_zodi_amplitude_prior(decomposer, kind, row_index):
    """Install this spectrum's geometry amplitude prior before fitting.

    The split-zodi priors are per-spectrum: the moon-share bracket and the
    absolute Leinert zodi bracket both come from a geometry prediction for this
    exposure and this telescope.  Only scalars are installed, so the design
    matrix is untouched and the basis stays identical for every row -- which is
    what lets the ML side reconstruct from coefficients with a single
    decomposer.

    Geometry that cannot be modelled (target below the horizon) clears the
    prior for that row instead of failing it: the fit then falls back to the
    shape bounds alone, which is exactly the pre-prior behaviour.

    The zodi total is recentred by SPLIT_ZODI_ZODI_PRIOR_CALIBRATION before it
    is installed; see that constant for the measurement.  The moon fraction is
    passed through untouched -- it is calibration-free by construction, so
    scaling it here would corrupt the one constraint that is not.
    """
    from skysub.sky_decomp.moon_zodi_model import (
        MoonZodiInvalidObservationError,
        geometry_amplitude_prior,
    )

    if _WORKER_META is None or not _WORKER_LSF:
        return
    _lsf = _sanitised_lsf_row(kind, row_index)
    if _lsf is None:
        # No usable LSF anywhere in this row: fall back to the shape bounds
        # alone, exactly as for geometry that cannot be modelled.
        decomposer.set_amplitude_prior(None, None)
        return
    try:
        fraction, zodi_total, _target_airmass = geometry_amplitude_prior(
            _WORKER_WAVE,
            _lsf,
            _moon_zodi_observation(kind, row_index),
            physical_to_fit_flux_scale=float(_WORKER_FACTOR),
            zodi_correction=SPLIT_ZODI_ZODI_CORRECTION,
        )
    except MoonZodiInvalidObservationError:
        decomposer.set_amplitude_prior(None, None)
        return
    decomposer.set_amplitude_prior(
        fraction, zodi_total * SPLIT_ZODI_ZODI_PRIOR_CALIBRATION
    )


def _science_line_mask_for_row(row_index):
    """Science-line mask for one row, centred on its measured Halpha velocity.

    Falls back to the worker's static rest-frame mask whenever centring is
    disabled, the science flux is unavailable, or the velocity cannot be
    measured -- ``measure_halpha_velocity`` returns 0.0 in that case, which
    reproduces the static mask exactly.  The window WIDTH never changes, so
    the number of masked pixels is constant to within a pixel or two and the
    dof count stays comparable from row to row.
    """
    if _WORKER_SCIENCE_LINE_MASK is None:
        return None
    if not _WORKER_SCIENCE_LINE_CENTRE:
        return _WORKER_SCIENCE_LINE_MASK
    sci = _WORKER_FLUX.get("sci")
    if sci is None:
        return _WORKER_SCIENCE_LINE_MASK
    sky = _WORKER_FLUX.get("sky1")
    if sky is None:
        sky = _WORKER_FLUX.get("sky2")
    velocity = measure_halpha_velocity(
        _WORKER_WAVE,
        np.asarray(sci[row_index], dtype=np.float64),
        None if sky is None else np.asarray(sky[row_index], dtype=np.float64),
    )
    if velocity == 0.0:
        return _WORKER_SCIENCE_LINE_MASK
    mask, _ = science_line_mask(
        _WORKER_WAVE,
        _WORKER_SCIENCE_LINE_FWHM,
        centre_velocity_km_s=velocity,
    )
    return mask


def _fit_ivar_row(kind, idx, flux_row):
    """Per-pixel ivar for one row: photon weights when enabled, else all ones.

    The all-ones fallback reproduces the historical behaviour exactly, so the
    flag is the only thing that changes a fit.
    """
    if not _WORKER_FIT_PIXEL_WEIGHTS or _WORKER_SENS_ABS is None:
        return np.ones_like(flux_row)
    from skysub.sky_decomp.pixel_weights import pixel_ivar

    n_fibres = None
    if _WORKER_META is not None:
        _col = FIT_PIXEL_WEIGHT_FIBRE_COLUMN.get(kind)
        # In the worker META is a numpy recarray (not an astropy Table), so ask
        # the dtype for the columns; accept either so the helper also works when
        # called from a notebook with a Table.
        _names = getattr(_WORKER_META, "colnames", None)
        if _names is None:
            _dt = getattr(_WORKER_META, "dtype", None)
            _names = getattr(_dt, "names", None) or ()
        if _col and _col in _names:
            _v = float(np.asarray(_WORKER_META[_col])[int(idx)])
            n_fibres = _v if np.isfinite(_v) and _v > 0.0 else None
    # `normalise=True` makes n_fibres, exptime and FACTOR cancel; they are passed
    # anyway so the call stays correct if the normalisation is ever turned off.
    return pixel_ivar(
        flux_row,
        _WORKER_WAVE,
        exptime=FIT_PIXEL_WEIGHT_EXPTIME_S,
        n_fibres=n_fibres,
        flux_scale=_WORKER_FACTOR,
        floor_frac=FIT_PIXEL_WEIGHT_FLOOR_FRAC,
        clip=_WORKER_FIT_PIXEL_WEIGHT_CLIP,
        sens=_WORKER_SENS_ABS,
        normalise=True,
    )


def _reliability_extra_meta(rows, n_rows):
    """Per-row reliability columns for the writer, or None if none were built.

    A row served from the compact cache carries no freshly computed flags, and a
    missing flag must not read as a CLEAN row: those get `reliability = -1`
    ("not evaluated") and NaN diagnostics, because the writer requires every row
    to carry the same columns.
    """
    present = [entry for entry in rows if entry is not None]
    if not present:
        return None
    filler = {
        key: (np.int32(-1) if key == "reliability" else float("nan"))
        for key in present[0]
    }
    return [dict(filler) if entry is None else entry for entry in rows[:n_rows]]


# Scalars that were INSTALLED on the decomposer for a row, read back so the
# binding tests compare against the bound the solve actually used rather than
# against a module default that a CLI flag may have overridden.
_CONSTRAINT_PRIOR_ATTRS = {
    "moon_fraction": "_amp_prior_moon_fraction",
    "zodi_total": "_amp_prior_zodi_total",
    "diffuse_oh_amp": "_diffuse_oh_amp",
    "amp_prior_tol": "amp_prior_tol",
    "amp_prior_floor": "amp_prior_floor",
    "zodi_amp_bound": "zodi_amp_bound",
    "diffuse_oh_centre_log10": "diffuse_oh_centre_log10",
    "diffuse_oh_bound_dex": "diffuse_oh_bound_dex",
    "diffuse_oh_relax_dex": "diffuse_oh_relax_dex",
    "diffuse_oh_gate_frac": "diffuse_oh_gate_frac",
    "diffuse_ratio_nominal": "diffuse_ratio_nominal",
    "diffuse_ratio_bound_dex": "diffuse_ratio_bound_dex",
    "moon_ratio_bound": "moon_ratio_bound",
    "zodi_ratio_bound": "zodi_ratio_bound",
}


def _constraint_prior(decomposer):
    if decomposer is None:
        return {}
    return {
        key: getattr(decomposer, attribute, None)
        for key, attribute in _CONSTRAINT_PRIOR_ATTRS.items()
    }


def _coef_blocks(result):
    """Moon and Zodi spline coefficients, for the adjacent-knot bound test."""
    names = [str(n) for n in (getattr(result, "design_names", None) or ())]
    coef = np.asarray(getattr(result, "coef", ()), dtype=np.float64).ravel()
    if not names or coef.size != len(names):
        return {}
    blocks = {}
    for family, prefix in (("moon", "Moon_bs"), ("zodi", "Zodi_bs")):
        index = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if index:
            blocks[family] = coef[np.asarray(index, dtype=int)]
    return blocks


def _row_colour_excess(row_index):
    """dC for this exposure, from all three arms of the input stack.

    A property of the ROW, not of the arm being fitted: the science fibre either
    carries continuum the sky model cannot represent or it does not, and the
    flag is set on every arm's row so a consumer joining on the row index sees
    it regardless of which product it is reading.
    """
    from skysub.sky_decomp import reliability as rel

    try:
        sci = _WORKER_FLUX["sci"][int(row_index)]
        near = _WORKER_FLUX["sky1"][int(row_index)]
        far = _WORKER_FLUX["sky2"][int(row_index)]
    except (KeyError, IndexError, TypeError):
        return float("nan")
    if _WORKER_WAVE is None:
        return float("nan")
    return rel.sci_colour_excess(sci, near, far, _WORKER_WAVE)


def _reliability_flags(result, retried=False, retry_bound=float("nan"),
                       was_reversed=False, decomposer=None, ivar=None,
                       row_index=None):
    """Per-row reliability columns for one finished fit.

    Computed here rather than in the writer because only the fitter knows
    whether a retry was run and which bounds were installed, and the result
    dataclasses use `slots=True` so nothing can be attached to them -- the
    columns travel as `extra_meta`.

    ERROR bits say the coefficients do not describe what their names say;
    WARNING bits say a constraint shaped the fit.  See `sky_decomp.reliability`.
    """
    from skysub.sky_decomp import reliability as rel

    bits = 0
    info = {"moon_slope": float("nan"), "zodi_slope": float("nan"),
            "separation": float("nan"), "moon_frac": float("nan")}
    constraint_info = {}
    excess = float("nan") if row_index is None else _row_colour_excess(row_index)
    if np.isfinite(excess) and excess > rel.SCI_COLOUR_EXCESS_MAX:
        bits |= rel.RELIABILITY_SCI_COLOUR_EXCESS
    if str(getattr(result, "fit_status", "")) != "Solved":
        bits |= rel.RELIABILITY_FIT_FAILED
    else:
        components = getattr(result, "components", None) or {}
        is_reversed, testable, info = rel.reversal_state(
            components, _WORKER_LOG_WAVE,
            min_component_frac=SPLIT_ZODI_REVERSAL_MIN_COMPONENT_FRAC,
            min_separation=SPLIT_ZODI_REVERSAL_MIN_SEPARATION)
        if is_reversed:
            bits |= rel.RELIABILITY_REVERSED
        if not testable:
            bits |= rel.RELIABILITY_REVERSAL_UNTESTABLE
        if rel.diffuse_collapsed(components,
                                 getattr(result, "bestfit_lsf", None)):
            bits |= rel.RELIABILITY_DIFFUSE_COLLAPSED
        if retried:
            bits |= rel.RELIABILITY_REVERSAL_RETRIED
            if was_reversed and not is_reversed:
                bits |= rel.RELIABILITY_REVERSAL_RECOVERED
        if decomposer is not None:
            good = None if ivar is None else (np.asarray(ivar) > 0.0)
            constraint_bits, constraint_info = rel.constraint_bits(
                components, _constraint_prior(decomposer), good=good,
                coef_blocks=_coef_blocks(result))
            bits |= constraint_bits
    return {
        "reliability": np.int32(bits),
        "reversal_separation": float(info["separation"]),
        "reversal_moon_frac": float(info["moon_frac"]),
        "reversal_retry_bound": float(retry_bound),
        "sci_colour_excess": float(excess),
        "moon_share": float(constraint_info.get("moon_share", float("nan"))),
        "zodi_int": float(constraint_info.get("zodi_int", float("nan"))),
        # The shape-bound BOOLEAN fires on ~100% of rows; this count (0 to 16 of
        # 18 adjacent pairs) is what distinguishes a lightly shaped fit from one
        # whose whole continuum sits on the bound.
        "shape_bound_pairs": np.int32(
            constraint_info.get("shape_bound_pairs", -1)),
        # UNCLAMPED target RMS of the seed and final-continuum solves. The
        # curvature penalties act at lambda * max(data_scale, 1)**2, so these
        # two columns are what let a run report its own effective smoothing
        # strength and its own clamp rate -- both of which move when the pixel
        # weighting changes, which is easy to miss because the nominal lambda
        # does not.
        "data_scale_seed": float(getattr(decomposer, "_data_scale_seed", np.nan)),
        "data_scale_cont": float(getattr(decomposer, "_data_scale_cont", np.nan)),
    }


def _fit_split_zodi_with_reversal_retry(decomposer, flux_row, ivar_row):
    """Fit; if the moon/zodi roles came out swapped, refit once tighter.

    Returns ``(result, retried, retry_bound, was_reversed)``.  The bounds are
    read by the solver through `getattr` at fit time (see
    `sky_decomp/fit.py`, the ratio_rows block), so tightening them needs no
    rebuild -- but the decomposer is reused for every row in this worker, so
    they are restored in a `finally`.
    """
    from skysub.sky_decomp import reliability as rel

    result = decomposer.fit(flux_row, ivar_row, verbose=False)
    if _WORKER_REVERSAL_RETRY_BOUND is None:
        return result, False, float("nan"), False
    if str(getattr(result, "fit_status", "")) != "Solved":
        return result, False, float("nan"), False
    is_reversed, _testable, _info = rel.reversal_state(
        getattr(result, "components", None) or {}, _WORKER_LOG_WAVE,
        min_component_frac=SPLIT_ZODI_REVERSAL_MIN_COMPONENT_FRAC,
        min_separation=SPLIT_ZODI_REVERSAL_MIN_SEPARATION)
    if not is_reversed:
        return result, False, float("nan"), False
    bound = float(_WORKER_REVERSAL_RETRY_BOUND)
    saved = (getattr(decomposer, "moon_ratio_bound", None),
             getattr(decomposer, "zodi_ratio_bound", None))
    try:
        decomposer.moon_ratio_bound = bound
        decomposer.zodi_ratio_bound = bound
        retry = decomposer.fit(flux_row, ivar_row, verbose=False)
    finally:
        if saved[0] is not None:
            decomposer.moon_ratio_bound = saved[0]
        if saved[1] is not None:
            decomposer.zodi_ratio_bound = saved[1]
    # A retry that fails to solve is worse than a reversed-but-solved row, so
    # keep the original and report it as reversed and not retried.
    if str(getattr(retry, "fit_status", "")) != "Solved":
        return result, False, float("nan"), True
    return retry, True, bound, True


def _fit_worker_row(kind, idx, flux_row, ivar_row):
    """Fit one row.  Returns ``(result, reliability_columns)``."""
    if _WORKER_SCIENCE_LINE_MASK is not None:
        ivar_row[_science_line_mask_for_row(idx)] = 0.0
    if _WORKER_FIT_MODEL == "baseline":
        result = _WORKER_DECOMPOSER.fit(
            flux_row, ivar_row, verbose=False, n_lsf_refits=3
        )
        return result, _reliability_flags(result, row_index=idx)
    if _WORKER_FIT_MODEL == "lsf-surface-iterative":
        result = _WORKER_DECOMPOSER.fit(flux_row, ivar_row, verbose=False)
        return result, _reliability_flags(result, row_index=idx)
    if _WORKER_FIT_MODEL in SPLIT_ZODI_FIT_MODELS:
        _install_split_zodi_amplitude_prior(_WORKER_DECOMPOSER, kind, idx)
        result, retried, bound, was_reversed = (
            _fit_split_zodi_with_reversal_retry(
                _WORKER_DECOMPOSER, flux_row, ivar_row))
        return result, _reliability_flags(
            result, retried, bound, was_reversed,
            decomposer=_WORKER_DECOMPOSER, ivar=ivar_row, row_index=idx)
    if _WORKER_FIT_MODEL == MOON_ZODI_FIT_MODEL:
        # Invalid source pixels remain on the native grid and are excluded only
        # by zero inverse variance.
        ivar_row = np.isfinite(flux_row).astype(np.float64)
        if _WORKER_SCIENCE_LINE_MASK is not None:
            ivar_row[_science_line_mask_for_row(idx)] = 0.0
        lsf_row = _sanitised_lsf_row(kind, idx)
        if lsf_row is None:
            # The physical predictor is driven by the detector LSF, so this
            # row cannot be fitted at all.  Record it as a failed input, the
            # same as an invalid airmass, rather than killing the chunk.
            reason = _unusable_lsf_reason(kind, idx)
            _report_unusable_lsf(kind, idx, reason)
            result = _WORKER_DECOMPOSER.failed_input_result(reason)
            return result, _reliability_flags(result, row_index=idx)
        result = _WORKER_DECOMPOSER.fit(
            flux_row,
            ivar_row,
            observation=_moon_zodi_observation(kind, idx),
            detector_lsf_fwhm=lsf_row,
            verbose=False,
        )
        return result, _reliability_flags(result, row_index=idx)
    if _WORKER_FIT_MODEL in TELLURIC_FIT_MODELS:
        try:
            decomposer = _telluric_decomposer(kind, idx)
        except _FailedInputError as error:
            schema = _telluric_decomposer(kind, idx, schema_only=True)
            result = schema.failed_input_result(str(error))
            return result, _reliability_flags(result, row_index=idx)
        if _WORKER_FIT_MODEL in SPLIT_ZODI_TELLURIC_FIT_MODELS:
            _install_split_zodi_amplitude_prior(decomposer, kind, idx)
            result, retried, bound, was_reversed = (
                _fit_split_zodi_with_reversal_retry(
                    decomposer, flux_row, ivar_row))
            return result, _reliability_flags(
                result, retried, bound, was_reversed,
                decomposer=decomposer, ivar=ivar_row, row_index=idx)
        result = decomposer.fit(flux_row, ivar_row, verbose=False)
        return result, _reliability_flags(
            result, decomposer=decomposer, ivar=ivar_row, row_index=idx)
    raise RuntimeError(f"Worker has unsupported fit model: {_WORKER_FIT_MODEL}")


def fit_chunk_worker(args):
    """Fit one chunk; compact mode caches rows and isolates row failures."""
    if _WORKER_DECOMPOSER is None:
        raise RuntimeError("Worker SkyDecomp has not been initialised.")
    kind, idx0, idx1 = args
    flux_chunk = np.asarray(_WORKER_FLUX[kind][idx0:idx1], dtype=np.float64)
    out = []
    for j in range(flux_chunk.shape[0]):
        idx = idx0 + j
        if _WORKER_COMPACT_CACHE_DIR is not None and _compact_cache_is_current(
            kind, idx
        ):
            out.append((idx, {"cached": True}, None))
        else:
            flux_row = flux_chunk[j] * _WORKER_FACTOR
            try:
                result, flags = _fit_worker_row(
                    kind, idx, flux_row, _fit_ivar_row(kind, idx, flux_row)
                )
            except Exception as error:
                if _WORKER_COMPACT_CACHE_DIR is None:
                    raise
                _save_compact_cache(kind, idx, error=error)
                out.append(
                    (idx,
                     {"cached": False, "error": f"{type(error).__name__}: {error}"},
                     None)
                )
            else:
                if _WORKER_COMPACT_CACHE_DIR is None:
                    out.append((idx, result, flags))
                else:
                    _save_compact_cache(kind, idx, result=result, flags=flags)
                    out.append((idx, {"cached": False}, flags))
        if _WORKER_PROGRESS_QUEUE is not None:
            _WORKER_PROGRESS_QUEUE.put(1)
    return kind, out


def resolve_base_dir(path_arg):
    """Accept either the project base dir or the palace dir and return SkyDecomp base_dir."""
    path = Path(path_arg).expanduser().resolve()

    if (path / "palace" / "PMD").exists():
        return path
    if path.name == "palace" and (path / "PMD").exists():
        return path.parent

    raise FileNotFoundError(
        "Could not resolve a valid SkyDecomp base directory from "
        f"{path}. Expected either a base dir containing palace/PMD or the palace directory itself."
    )


def resolve_runtime_data_roots(
    fit_model,
    palace_dir=None,
    moon_zodi_data_root=None,
):
    """Resolve only the data contract used by the selected fit model."""
    if fit_model == MOON_ZODI_FIT_MODEL or fit_model in TELLURIC_FIT_MODELS:
        candidate = (
            moon_zodi_data_root
            if moon_zodi_data_root is not None
            else palace_dir
            if palace_dir is not None
            else DEFAULT_MOON_ZODI_DATA_ROOT
        )
        data_root = Path(candidate).expanduser().resolve()
        validate_decomposition_data_root(str(data_root))
        return data_root, data_root

    if fit_model in SPLIT_ZODI_FIT_MODELS and (
        moon_zodi_data_root is not None or palace_dir is None
    ):
        candidate = (
            moon_zodi_data_root
            if moon_zodi_data_root is not None
            else DEFAULT_MOON_ZODI_DATA_ROOT
        )
        data_root = Path(candidate).expanduser().resolve()
        validate_decomposition_data_root(str(data_root))
        return data_root, data_root

    if fit_model in SPLIT_ZODI_FIT_MODELS and palace_dir is not None:
        candidate = Path(palace_dir).expanduser().resolve()
        if (candidate / "bundle_manifest.json").is_file():
            validate_decomposition_data_root(str(candidate))
            return candidate, candidate

    if palace_dir is None:
        raise ValueError(
            "palace_dir is required for baseline and non-split "
            "lsf-surface-iterative fits"
        )
    return resolve_base_dir(palace_dir), None


def _iter_chunk_tasks(n_rows, chunk_size):
    for kind in ("sci", "sky1", "sky2"):
        for i0 in range(0, n_rows, chunk_size):
            i1 = min(i0 + chunk_size, n_rows)
            # Sending only index ranges keeps per-task IPC tiny.
            yield (kind, i0, i1)


def _compact_run_provenance(data_file, wave, fit_model, base_dir, parameters):
    source_dir = Path(__file__).resolve().parent / "sky_decomp"
    payload = {
        "schema_version": 1,
        "storage": "coefficient-and-continuous-lsf-per-row-npz-v1",
        "input_path": str(Path(data_file).resolve()),
        "input_sha256": file_sha256(data_file),
        "native_wave_pixels": int(np.asarray(wave).size),
        "wave_sha256": wave_sha256(wave),
        "fit_model": fit_model,
        "parameters": parameters,
        "bundle_manifest_sha256": file_sha256(Path(base_dir) / "bundle_manifest.json"),
        "source_sha256": {"decompose_parallel.py": file_sha256(Path(__file__))}
        | {
            name: file_sha256(source_dir / name)
            for name in (
                "fit.py",
                "lsf_spline2d.py",
                "lsf_surface_iterative.py",
                "residual_pca.py",
                "telluric_corrected_lines.py",
            )
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_default).encode()
    ).hexdigest()
    return payload | {"run_fingerprint": fingerprint}


def _worker_initargs(
    *, wave, lsf_sigma, base_dir, factor, data_file, progress_queue, fit_model,
    n_refinement_cycles, worker_counter, pin_workers, diagnose_threads,
    palace_suffix, palace_oh_suffix, palace_diffuse_suffix, exposure_seconds,
    moon_zodi_data_root, n_spline_knots, n_zodi_spline_knots, zodi_smooth_lambda,
    moon_smooth_lambda, mask_science_lines, centre_on_halpha, fit_pixel_weights,
    fit_pixel_weight_clip, reversal_retry_bound, diffuse_ratio_bound_dex,
    diffuse_ratio_nominal, diffuse_oh_centre_log10, diffuse_oh_bound_dex,
    compact_cache_root, run_fingerprint,
):
    """The positional `init_worker` arguments of a run -- one definition.

    Shared by `run` (the process pool) and `decompose_in_process` (a single
    exposure in the caller's process), so the two can never set a worker up
    differently.
    """
    return (
        wave,
        lsf_sigma,
        str(base_dir),
        float(factor),
        data_file if isinstance(data_file, fits.HDUList) else str(data_file),
        progress_queue,
        fit_model,
        n_refinement_cycles,
        worker_counter,
        bool(pin_workers),
        bool(diagnose_threads),
        palace_suffix,
        palace_oh_suffix,
        palace_diffuse_suffix,
        exposure_seconds,
        None if moon_zodi_data_root is None else str(moon_zodi_data_root),
        int(n_spline_knots),
        int(n_zodi_spline_knots),
        float(zodi_smooth_lambda),
        float(moon_smooth_lambda),
        bool(mask_science_lines),
        bool(centre_on_halpha),
        bool(fit_pixel_weights),
        None if fit_pixel_weight_clip is None else float(fit_pixel_weight_clip),
        None if reversal_retry_bound is None else float(reversal_retry_bound),
        float(diffuse_ratio_bound_dex),
        diffuse_ratio_nominal,
        diffuse_oh_centre_log10,
        float(diffuse_oh_bound_dex),
        None if compact_cache_root is None else str(compact_cache_root),
        run_fingerprint,
    )


def run(
    data_file,
    palace_dir,
    n_workers,
    lsf_sigma,
    factor,
    output_dir,
    chunk_size,
    max_in_flight,
    fit_model="baseline",
    n_refinement_cycles=5,
    limit=None,
    pin_workers=False,
    diagnose_threads=False,
    palace_suffix=None,
    palace_oh_suffix=None,
    palace_diffuse_suffix=None,
    exposure_seconds=900.0,
    moon_zodi_data_root=None,
    n_spline_knots=MOON_N_KNOTS_DEFAULT,
    n_zodi_spline_knots=SPLIT_ZODI_N_KNOTS_DEFAULT,
    zodi_smooth_lambda=SPLIT_ZODI_SMOOTH_LAMBDA_DEFAULT,
    moon_smooth_lambda=MOON_SMOOTH_LAMBDA_DEFAULT,
    diffuse_ratio_bound_dex=SPLIT_ZODI_DIFFUSE_RATIO_BOUND_DEX,
    diffuse_ratio_nominal=SPLIT_ZODI_DIFFUSE_RATIO_NOMINAL,
    diffuse_oh_centre_log10=SPLIT_ZODI_DIFFUSE_OH_CENTRE_LOG10,
    diffuse_oh_bound_dex=SPLIT_ZODI_DIFFUSE_OH_BOUND_DEX,
    mask_science_lines=SCIENCE_LINE_MASK_ENABLED,
    centre_on_halpha=SCIENCE_LINE_MASK_CENTRE_ON_HALPHA,
    fit_pixel_weights=FIT_PIXEL_WEIGHTS,
    fit_pixel_weight_clip=FIT_PIXEL_WEIGHT_CLIP,
    reversal_retry_bound=(
        SPLIT_ZODI_REVERSAL_RETRY_BOUND if SPLIT_ZODI_REVERSAL_RETRY else None
    ),
    compact_only=False,
):
    palace_oh_suffix = _resolved_palace_oh_suffix(fit_model, palace_oh_suffix)
    base_dir, resolved_moon_zodi_data_root = resolve_runtime_data_roots(
        fit_model,
        palace_dir=palace_dir,
        moon_zodi_data_root=moon_zodi_data_root,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_file} ...")
    wave = fits.getdata(data_file, "WAVE").astype(np.float64)
    with fits.open(data_file, memmap=True) as hdul:
        n_rows_total = int(hdul["FLUX_SCI"].data.shape[0])

    n_rows = n_rows_total if limit is None else min(n_rows_total, int(limit))

    compact_provenance = compact_cache_root = run_fingerprint = None
    if compact_only:
        parameters = {
            "factor": factor,
            "lsf_sigma": lsf_sigma,
            "n_refinement_cycles": n_refinement_cycles,
            "n_spline_knots": n_spline_knots,
            "n_zodi_spline_knots": n_zodi_spline_knots,
            "zodi_smooth_lambda": zodi_smooth_lambda,
            "diffuse_ratio_bound_dex": diffuse_ratio_bound_dex,
            "diffuse_ratio_nominal": diffuse_ratio_nominal,
            "diffuse_oh_centre_log10": diffuse_oh_centre_log10,
            "diffuse_oh_bound_dex": diffuse_oh_bound_dex,
            "mask_science_lines": mask_science_lines,
            "centre_on_halpha": centre_on_halpha,
            "fit_pixel_weights": fit_pixel_weights,
            "fit_pixel_weight_clip": fit_pixel_weight_clip,
            "reversal_retry_bound": reversal_retry_bound,
            "palace_suffix": palace_suffix,
            "palace_oh_suffix": palace_oh_suffix,
            "palace_diffuse_suffix": palace_diffuse_suffix,
            # A resumed run must not mix rows anchored to different zodi models.
            "zodi_correction": SPLIT_ZODI_ZODI_CORRECTION,
        }
        compact_provenance = _compact_run_provenance(
            data_file, wave, fit_model, base_dir, parameters
        )
        run_fingerprint = compact_provenance["run_fingerprint"]
        stem = Path(data_file).stem
        suffix = FIT_MODEL_SUFFIXES[fit_model]
        compact_cache_root = output_dir / f".{stem}{suffix}_compact_cache"
        compact_cache_root.mkdir(parents=True, exist_ok=True)
        (compact_cache_root / "run_provenance.json").write_text(
            json.dumps(compact_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if limit is not None and n_rows < n_rows_total:
        print(f"  {n_rows} spectra (limited from {n_rows_total}), {len(wave)} wavelength pixels")
    else:
        print(f"  {n_rows} spectra, {len(wave)} wavelength pixels")
    print(f"  n_workers={n_workers}, lsf_sigma={lsf_sigma}, factor={factor}")
    print(f"  chunk_size={chunk_size}, max_in_flight={max_in_flight}")
    print(f"  fit_model={fit_model}, n_refinement_cycles={n_refinement_cycles}")
    print(f"  palace_suffix={palace_suffix!r}")
    print(f"  palace_oh_suffix={palace_oh_suffix!r}")
    print(f"  palace_diffuse_suffix={palace_diffuse_suffix!r}")
    print(f"  exposure_seconds_fallback={exposure_seconds}")
    if resolved_moon_zodi_data_root is not None:
        print(f"  bundled_data_root={str(resolved_moon_zodi_data_root)!r}")
    if fit_model in (
        "lsf-surface-iterative",
        *SPLIT_ZODI_FIT_MODELS,
        *TELLURIC_FIT_MODELS,
    ):
        print(f"  n_spline_knots={n_spline_knots} "
              f"(Moon_bs basis = {int(n_spline_knots) + 4})")
    if fit_model in SPLIT_ZODI_FIT_MODELS:
        print(f"  n_zodi_spline_knots={n_zodi_spline_knots}, "
              f"zodi_smooth_lambda={zodi_smooth_lambda}")
    # Printed unconditionally: these two together decide the smoothing every
    # row actually receives, and the pair is what a later A/B has to match.
    print(f"  moon_smooth_lambda={moon_smooth_lambda}")
    print(f"  pin_workers={pin_workers}, diagnose_threads={diagnose_threads}")
    print(f"  base_dir={base_dir}")

    primary_meta = _fit_model_primary_meta(fit_model, palace_oh_suffix)
    n_tasks = int(np.ceil(n_rows / chunk_size)) * 3
    results = {kind: [None] * n_rows for kind in ("sci", "sky1", "sky2")}
    reliability = {kind: [None] * n_rows for kind in ("sci", "sky1", "sky2")}
    completed = 0

    t0 = time.perf_counter()

    # spawn everywhere: `fork` inherits parent BLAS pools and undermines thread limits.
    mp_context = mp.get_context("spawn")
    progress_queue = mp_context.Queue()
    worker_counter = mp_context.Value("i", 0)

    def _drain_progress_queue():
        increment = 0
        while True:
            try:
                increment += progress_queue.get_nowait()
            except queue_mod.Empty:
                break
        if increment:
            pbar.update(increment)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_context,
        initializer=init_worker,
        initargs=_worker_initargs(
            wave=wave, lsf_sigma=lsf_sigma, base_dir=base_dir, factor=factor,
            data_file=data_file, progress_queue=progress_queue,
            fit_model=fit_model, n_refinement_cycles=n_refinement_cycles,
            worker_counter=worker_counter, pin_workers=pin_workers,
            diagnose_threads=diagnose_threads, palace_suffix=palace_suffix,
            palace_oh_suffix=palace_oh_suffix,
            palace_diffuse_suffix=palace_diffuse_suffix,
            exposure_seconds=exposure_seconds,
            moon_zodi_data_root=resolved_moon_zodi_data_root,
            n_spline_knots=n_spline_knots,
            n_zodi_spline_knots=n_zodi_spline_knots,
            zodi_smooth_lambda=zodi_smooth_lambda,
            moon_smooth_lambda=moon_smooth_lambda,
            mask_science_lines=mask_science_lines,
            centre_on_halpha=centre_on_halpha,
            fit_pixel_weights=fit_pixel_weights,
            fit_pixel_weight_clip=fit_pixel_weight_clip,
            reversal_retry_bound=reversal_retry_bound,
            diffuse_ratio_bound_dex=diffuse_ratio_bound_dex,
            diffuse_ratio_nominal=diffuse_ratio_nominal,
            diffuse_oh_centre_log10=diffuse_oh_centre_log10,
            diffuse_oh_bound_dex=diffuse_oh_bound_dex,
            compact_cache_root=compact_cache_root,
            run_fingerprint=run_fingerprint,
        ),
    ) as executor:
        pbar = tqdm(
            total=3 * n_rows,
            desc="decomp",
            unit=" decomp",
            mininterval=0.2,
            position=0,
            leave=True,
        )
        pbar.set_postfix(chunks=f"0/{n_tasks}")
        pbar.refresh()

        task_iter = iter(_iter_chunk_tasks(n_rows, chunk_size))
        pending = set()

        def _submit_until_full():
            while len(pending) < max_in_flight:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return
                pending.add(executor.submit(fit_chunk_worker, task))

        _submit_until_full()
        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            _drain_progress_queue()
            for future in done:
                kind, chunk_results = future.result()
                for idx, result, flags in chunk_results:
                    results[kind][idx] = result
                    if flags is not None:
                        reliability[kind][idx] = flags
                completed += 1
                pbar.set_postfix(chunks=f"{completed}/{n_tasks}")
            _submit_until_full()
        _drain_progress_queue()
        pbar.close()

    progress_queue.close()
    progress_queue.join_thread()

    elapsed = time.perf_counter() - t0
    print(f"Fitting done in {elapsed:.1f}s ({elapsed / n_rows:.2f}s per spectrum)")

    stem = Path(data_file).stem
    suffix = FIT_MODEL_SUFFIXES[fit_model]
    for kind in ("sci", "sky1", "sky2"):
        if compact_only:
            _write_compact_fits(
                compact_cache_root,
                kind,
                n_rows,
                run_fingerprint,
                fit_model,
                output_dir / f"{stem}_{kind}_meta_coef{suffix}.fits",
                primary_meta=primary_meta,
            )
        else:
            results_to_fits(
                results[kind],
                output_dir / f"{stem}_decomp_{kind}{suffix}.fits",
                extra_meta=_reliability_extra_meta(reliability[kind], n_rows),
                primary_meta=primary_meta,
            )
    return compact_provenance


def _copy_hdu_with_name(hdu, extname):
    """Return a copy of an HDU with a new extension name."""
    header = hdu.header.copy()
    header["EXTNAME"] = extname
    return type(hdu)(data=hdu.data, header=header, name=extname)


def _infer_decomp_label(path, index):
    """Infer a stable label for a decomposition file from its filename."""
    name = Path(path).name.lower()
    for label in ("sky1", "sky2", "sci"):
        if label in name:
            return label.upper()
    return f"DEC{index}"


def extract_meta_and_coef_products(
    input_fits_path,
    decomp_fits_path_1,
    decomp_fits_path_2,
    decomp_fits_path_3,
    meta_output_path=None,
    sky1_output_path=None,
    sky2_output_path=None,
    sci_output_path=None,
):
    """Write compact FITS products containing only selected extensions.

    The first output contains only the META extension from `input_fits_path`.
    Each decomposition input gets its own output FITS containing META and COEF.
    Extended LSF-surface products are copied when present. Default output
    paths are written in the current working directory.
    """
    input_path = Path(input_fits_path)
    cwd = Path.cwd()
    if meta_output_path is None:
        meta_output_path = str(cwd / f"{input_path.stem}_meta_only{input_path.suffix}")

    decomp_files = [decomp_fits_path_1, decomp_fits_path_2, decomp_fits_path_3]
    decomp_outputs = [sky1_output_path, sky2_output_path, sci_output_path]

    with fits.open(input_fits_path) as hdul_in:
        if "META" not in hdul_in:
            raise KeyError(f"Missing META extension in {input_fits_path}")
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                _copy_hdu_with_name(hdul_in["META"], "META"),
            ]
        ).writeto(meta_output_path, overwrite=True)

    resolved_outputs = []
    for index, decomp_path in enumerate(decomp_files, start=1):
        label = _infer_decomp_label(decomp_path, index)
        out_path = decomp_outputs[index - 1]
        if out_path is None:
            stem_lower = Path(decomp_path).stem.lower()
            if "lsf_spline2d_split_zodi" in stem_lower:
                variant = "_lsf_spline2d_split_zodi"
            elif "moon_zodi_lsf_surface_iterative" in stem_lower:
                variant = "_moon_zodi_lsf_surface_iterative"
            elif "lsf_surface_iterative" in stem_lower:
                variant = "_lsf_surface_iterative"
            else:
                variant = ""
            out_path = str(
                cwd / (f"{input_path.stem}_{label.lower()}_meta_coef{variant}{input_path.suffix}")
            )
        with fits.open(decomp_path) as hdul_dec:
            for extname in ("META", "COEF"):
                if extname not in hdul_dec:
                    raise KeyError(f"Missing {extname} extension in {decomp_path}")
            compact_hdus = [
                fits.PrimaryHDU(header=hdul_dec[0].header.copy()),
                _copy_hdu_with_name(hdul_dec["META"], "META"),
                _copy_hdu_with_name(hdul_dec["COEF"], "COEF"),
            ]
            if "COEF_ERR" in hdul_dec:
                compact_hdus.append(_copy_hdu_with_name(hdul_dec["COEF_ERR"], "COEF_ERR"))
            for cov_name in ("COEF_COV_MOON", "COEF_COV_ZODI"):
                if cov_name in hdul_dec:
                    compact_hdus.append(_copy_hdu_with_name(hdul_dec[cov_name], cov_name))
            lsf_extensions = ("LSF_COEF", "LSF_KNOTS", "LSF_META")
            present = [name in hdul_dec for name in lsf_extensions]
            if any(present) and not all(present):
                raise KeyError(f"Incomplete LSF-surface extensions in {decomp_path}")
            if all(present):
                compact_hdus.extend(
                    _copy_hdu_with_name(hdul_dec[name], name) for name in lsf_extensions
                )
            moon_zodi_extensions = ("MZ_MODEL", "MZ_ASSETS", "MZ_KNOTS", "MZ_META")
            moon_zodi_present = [name in hdul_dec for name in moon_zodi_extensions]
            if any(moon_zodi_present) and not all(moon_zodi_present):
                raise KeyError(f"Incomplete Moon/Zodi extensions in {decomp_path}")
            if all(moon_zodi_present):
                compact_hdus.extend(
                    _copy_hdu_with_name(hdul_dec[name], name)
                    for name in moon_zodi_extensions
                )
            fits.HDUList(compact_hdus).writeto(out_path, overwrite=True)
        print(f"Wrote {label} META/COEF file -> {out_path}")
        resolved_outputs.append(out_path)

    print(f"Wrote META-only file -> {meta_output_path}")
    return (meta_output_path, *resolved_outputs)

def thin_fits_every_n(input_path, output_path, n, row_hdu_name="META"):
    """Write a new FITS with every n-th row-like element kept.

    The function preserves HDU structure and headers. It identifies the row
    count from `row_hdu_name` (default: META), then slices any table HDU with
    that row count and any image HDU whose first axis matches that row count.

    Tables that reference the row axis indirectly through a `spectrum_index`
    column (such as `LSF_META`, which has one row per channel per spectrum)
    are filtered to rows whose `spectrum_index` is in the kept set, and the
    `spectrum_index` values are remapped to the new 0-based row positions so
    downstream loaders can still index a thinned coefficient cube directly.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    with fits.open(input_path) as hdul:
        if row_hdu_name not in hdul:
            raise KeyError(f"HDU '{row_hdu_name}' not found in {input_path}")

        n_rows = len(hdul[row_hdu_name].data)
        indices = np.arange(n_rows, dtype=int)[::n]
        remap = {int(old): int(new) for new, old in enumerate(indices)}
        keep = slice(None, None, n)
        global_hdus = {"MZ_MODEL", "MZ_ASSETS", "MZ_KNOTS"}

        out_hdus = []
        for hdu in hdul:
            header = hdu.header.copy()

            if isinstance(hdu, fits.PrimaryHDU):
                data = hdu.data
                if data is not None and getattr(data, "ndim", 0) >= 1 and data.shape[0] == n_rows:
                    data = data[keep, ...]
                out_hdus.append(fits.PrimaryHDU(data=data, header=header))

            elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                data = hdu.data
                if hdu.name == "LSF_META" and data is not None:
                    selected = np.isin(np.asarray(data["spectrum_index"], dtype=int), indices)
                    data = data[selected].copy()
                    for row in data:
                        row["spectrum_index"] = remap[int(row["spectrum_index"])]
                elif hdu.name == "MZ_META" and data is not None:
                    data = data[keep].copy()
                    data["spectrum_index"] = np.arange(len(data), dtype=int)
                elif hdu.name not in global_hdus and data is not None and len(data) == n_rows:
                    data = data[keep]
                elif (
                    data is not None
                    and "spectrum_index" in data.dtype.names
                    and len(data) % n_rows == 0
                ):
                    # Multi-row-per-spectrum table (e.g. LSF_META has one row
                    # per channel per spectrum). Filter by spectrum_index and
                    # remap to the thinned cube's 0-based positions.
                    si = np.asarray(data["spectrum_index"], dtype=np.int64)
                    mask = np.isin(si, indices)
                    data = data[mask].copy()
                    remapped = np.fromiter(
                        (remap[int(v)] for v in data["spectrum_index"]),
                        dtype=np.int64,
                        count=len(data),
                    )
                    data["spectrum_index"] = remapped
                out_hdus.append(type(hdu)(data=data, header=header, name=hdu.name))

            elif isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                data = hdu.data
                if (
                    hdu.name not in global_hdus
                    and data is not None
                    and getattr(data, "ndim", 0) >= 1
                    and data.shape[0] == n_rows
                ):
                    data = data[keep, ...]
                out_hdus.append(type(hdu)(data=data, header=header, name=hdu.name))

            else:
                out_hdus.append(hdu.copy())

        fits.HDUList(out_hdus).writeto(output_path, overwrite=True)


def build_arg_parser():
    """The command-line parser; its defaults ARE a default cluster run."""
    parser = argparse.ArgumentParser(description="LVM sky spectral decomposition")
    parser.add_argument("data_file", help="Input FITS file (median stacked LVM frame)")
    parser.add_argument(
        "palace_dir",
        nargs="?",
        default=None,
        help=(
            "Legacy project/PALACE root, or a complete bundled data root containing "
            "bundle_manifest.json. Required by baseline and non-split "
            "lsf-surface-iterative; optional for bundled production modes."
        ),
    )
    parser.add_argument(
        "--n-workers", type=int, default=4, help="Number of parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--lsf-sigma", type=float, default=0.5, help="LSF Gaussian sigma in Å (default: 0.5)"
    )
    parser.add_argument(
        "--factor", type=float, default=1e14, help="Flux scaling factor (default: 1e14)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64, help="Rows per worker task chunk (default: 64)"
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Max submitted chunks waiting/running at once (default: n-workers)",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result FITS files (default: .)"
    )
    parser.add_argument(
        "--fit-model",
        # 2026-09-18: the non-telluric fit models are no longer offered.  The
        # telluric fit is the supported decomposition on this branch -- it fits
        # ~30% better full-band and every downstream path (reconstruction,
        # wavelength cache, diagnostics) now assumes it.  The implementations
        # are still present because the telluric classes INHERIT from them
        # (SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D -> ... ->
        # SkyDecompLSFSurfaceIterative -> SkyDecomp), so they are shared
        # machinery rather than an alternative path; they are simply no longer
        # reachable from the CLI.  To reproduce a pre-2026-09-17 corpus, check
        # out a commit from before this change.
        choices=tuple(TELLURIC_FIT_MODELS),
        # 2026-09-24: the SkyFar ridge-corrected PALACE OH line strengths.
        # Same 388 coefficients (OH coefs move <1% at p90); on a 1000-row A/B
        # OH-pixel chi2 drops x0.03 (r) / x0.28 (z), full band x0.69, with the
        # continuum partition and reliability flags unchanged.  Outputs carry
        # the _palacecorr_ suffix.
        default=PALACECORR_VNF_SPLIT_ZODI_FIT_MODEL,
        help="Fit implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--n-refinement-cycles",
        type=int,
        default=5,
        help="Continuum/LSF/line cycles for iterative LSF methods (default: 5)",
    )
    parser.add_argument(
        "--palace-suffix",
        default=None,
        help=(
            "Optional suffix for versioned pmd_popmodel_OH and pmd_refcont files. "
            "The suffix is appended exactly; for example, '_adam_v1' selects "
            "pmd_popmodel_OH_adam_v1.dat and "
            "pmd_refcont_adam_v1.dat (legacy default: canonical unsuffixed "
            "files; bundled Moon/Zodi modes use their manifest defaults)."
        ),
    )
    parser.add_argument(
        "--palace-oh-suffix",
        default=None,
        help="Optional exact suffix overriding --palace-suffix for pmd_popmodel_OH only.",
    )
    parser.add_argument(
        "--palace-diffuse-suffix",
        default=None,
        help="Optional exact suffix overriding --palace-suffix for pmd_refcont only.",
    )
    parser.add_argument(
        "--exposure-seconds",
        type=float,
        default=900.0,
        help=(
            "Exposure duration used only when META has no exposure duration "
            "(v1 provenance contract requires the default 900 s assumption)."
        ),
    )
    parser.add_argument(
        "--moon-zodi-data-root",
        type=Path,
        default=None,
        help=(
            "Complete data root containing moon_zodi/ and palace/PMD for the "
            "Moon/Zodi, split-zodi, or telluric method (default: packaged "
            "skysub/sky_decomp/data)."
        ),
    )
    parser.add_argument(
        "--n-spline-knots",
        type=int,
        default=MOON_N_KNOTS_DEFAULT,
        help=(
            f"Interior B-spline knots for Moon_bs when --fit-model is "
            f"lsf-surface-iterative, split-zodi, or telluric "
            f"(default: {MOON_N_KNOTS_DEFAULT}; Moon_bs basis = knots + 4)."
        ),
    )
    parser.add_argument(
        "--n-zodi-spline-knots",
        type=int,
        default=SPLIT_ZODI_N_KNOTS_DEFAULT,
        help=(
            f"Interior B-spline knots for Zodi_bs with a split-zodi method "
            f"(default: {SPLIT_ZODI_N_KNOTS_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--zodi-smooth-lambda",
        type=float,
        default=SPLIT_ZODI_SMOOTH_LAMBDA_DEFAULT,
        help=(
            f"Curvature penalty on Zodi_bs with a split-zodi method "
            f"(default: {SPLIT_ZODI_SMOOTH_LAMBDA_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--moon-smooth-lambda",
        type=float,
        default=MOON_SMOOTH_LAMBDA_DEFAULT,
        help=(
            f"Curvature penalty on Moon_bs (default: {MOON_SMOOTH_LAMBDA_DEFAULT}). "
            "This is what EVERY row gets: the penalty is normalised by "
            "data_scale**2, so it means the same thing on a bright row as on "
            "a faint one. Before 2026-09-20 it acted as a per-row floor that "
            "the brightest rows exceeded by up to ~3200x, so values tuned "
            "against older corpora do not carry over."
        ),
    )
    parser.add_argument(
        "--diffuse-ratio-bound-dex",
        type=float,
        default=SPLIT_ZODI_DIFFUSE_RATIO_BOUND_DEX,
        help=(
            "Half-width in dex of the bracket on the diffuse species ratios "
            "FeO/HO2 and O2Ac/HO2, about --diffuse-ratio-nominal. 0 disables. "
            "The three species are individually unidentifiable in the LVM band "
            "-- the three arms of one exposure disagree by 0.54 dex at the "
            "median -- so this removes freedom the data cannot measure. "
            "Requires --diffuse-ratio-nominal."
        ),
    )
    parser.add_argument(
        "--diffuse-oh-bound-dex",
        type=float,
        default=SPLIT_ZODI_DIFFUSE_OH_BOUND_DEX,
        help=(
            "Half-width in dex of the MOON-GATED upper bound on the WHOLE "
            "diffuse block, (A_HO2 + A_FeO + A_O2Ac) / A_OH, above "
            "--diffuse-oh-centre-log10. 0 disables. The species are mesospheric "
            "and cannot depend on the moon, but the block's amplitude relative "
            "to OH rises a factor 4.5 with moon_frac_po, so the templates are "
            "absorbing scattered moonlight. One-sided and gated because the "
            "dark-time scatter of the ratio is 0.306 dex and real. The block, "
            "not FeO alone: an FeO-only cap left 195 of 415 gated rows above "
            "their bound because the three species move together. Went 0.15 -> "
            "0.30 -> 0.15 over 2026-09-11/12: the OH regression that motivated "
            "0.30 was a degenerate-metric artefact (mesospheric COEFFICIENT "
            "gain -3.9%% but the same errors in FLUX space +4.9%%), while 0.30 "
            "gave back half the leak suppression (FeO/OH moon Q4/Q1 1.33x -> "
            "2.16x against 3.44x uncapped). See the block comment at "
            "SPLIT_ZODI_DIFFUSE_OH_BOUND_DEX."
        ),
    )
    parser.add_argument(
        "--diffuse-oh-centre-log10",
        type=float,
        default=SPLIT_ZODI_DIFFUSE_OH_CENTRE_LOG10,
        help=(
            "log10 of the (A_HO2 + A_FeO + A_O2Ac) / A_OH centre for the bound "
            "above: the DARK-TIME median over diffuse-live rows, where there is "
            "no moon to leak. Re-measure per corpus -- but note the stored OH "
            "coefficients are on the convolved STICK basis, not matrix_oh, so "
            "integrate the stored COMP_* planes rather than assembling the "
            "ratio from matrix_*.sum(axis=1), which is 5.01x wrong on the OH "
            "side. Confirmed -0.6489 against a measured -0.6616 on "
            "gaia-stars-mask-cont."
        ),
    )
    parser.add_argument(
        "--diffuse-ratio-nominal",
        type=str,
        default=",".join(str(v) for v in SPLIT_ZODI_DIFFUSE_RATIO_NOMINAL),
        help=(
            "Comma-separated HO2,FeO,O2Ac FLUX shares giving the centre of the "
            "ratio bracket, e.g. '0.0396,0.7026,0.2578'. MEASURE THIS ON THE "
            "CORPUS BEING FITTED: centring on PALACE's own shares costs 25.7%% "
            "of the blue chi2 against 0.67%% for the corpus median, because "
            "PALACE is calibrated for Paranal and LVM observes from LCO."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input rows (default: process all rows)",
    )
    parser.add_argument(
        "--pin-workers",
        action="store_true",
        help=(
            "Pin each worker to a single CPU core via sched_setaffinity (Linux). "
            "Nuclear option: hard-caps observed CPU usage per worker to 100%% even if a "
            "library ignores thread-limit env vars."
        ),
    )
    parser.add_argument(
        "--diagnose-threads",
        action="store_true",
        help="Print per-library thread pool counts from worker 0 after all imports finish.",
    )
    parser.add_argument(
        "--no-science-line-mask",
        action="store_true",
        help=(
            "Fit the science emission-line windows instead of excluding them. "
            "By default [OII]3727, Hbeta, [OIII]4959/5007, Halpha, [NII]6548/6583 "
            "and [SII]6716/6731 are masked via IVAR=0 in every arm, because none "
            "of them is in the basis and the fitted OH component absorbs them "
            "(measured 6.7x the sideband OH level inside Halpha/[NII] on an "
            "inner-plane H II region field).  Use this to A/B the mask."
        ),
    )
    parser.add_argument(
        "--no-fit-pixel-weights",
        action="store_true",
        help=(
            "Fit with ivar = 1 on every pixel (the historical unweighted mask) "
            "instead of the absolute photon model, which is now the DEFAULT. "
            "The default weighting uses the same sensitivity curves and "
            "variance floor as the ML loss, vendored in "
            "sky_decomp/data/sensitivity, row-normalised to mean 1 so exptime, "
            "the fibre count and FACTOR cancel and every tuned regularisation "
            "constant keeps its meaning; measured on 1447 rows it improves the "
            "absolute photon chi2 by 11-14%% full band and 2-4%% in the blue "
            "for -0.62%% yield through the diffuse-collapse gate. Use this flag "
            "to reproduce a pre-2026-09-18 corpus, remembering that "
            "reduced_chi2 is a weighted chi2 with the weights on and an "
            "unweighted residual-per-pixel with them off, so its values are "
            "not comparable between the two. See FIT_PIXEL_WEIGHTS."
        ),
    )
    parser.add_argument(
        "--fit-pixel-weight-clip",
        type=float,
        default=FIT_PIXEL_WEIGHT_CLIP,
        metavar="FACTOR",
        help=(
            "Bound the fit weights to [1/FACTOR, FACTOR] about the row mean "
            "(ignored with --no-fit-pixel-weights). The raw photon weights span "
            "~600x within a row, enough for the fit to abandon a faint blue "
            "component: FACTOR 3 stopped the one diffuse collapse seen in 20 "
            "rows and kept half the chi2 gain (-5.8%% full band). Default: "
            "unclipped."
        ),
    )
    parser.add_argument(
        "--no-reversal-retry",
        action="store_true",
        help=(
            "Do not refit moon/zodi role-reversed rows with a tighter spline "
            "shape bound. A reversal is a shape-labelling artefact, not a "
            "brightness error, and the corpus build has simply DROPPED such "
            "rows (2.3%% of every10; ~220 on the full corpus) -- which "
            "production cannot do. Measured on 18 reversed rows, all 18 "
            "recover at bound 0.85 for +2.8%% median chi2 with the amplitudes "
            "essentially unchanged. Retried rows are flagged in the "
            "reliability column either way. See SPLIT_ZODI_REVERSAL_RETRY."
        ),
    )
    parser.add_argument(
        "--reversal-retry-bound",
        type=float,
        default=SPLIT_ZODI_REVERSAL_RETRY_BOUND,
        metavar="BETA",
        help=(
            "Moon/zodi adjacent-knot ratio bound used for the reversal retry "
            "only (default %(default)s); every row that is not reversed keeps "
            "the deployed SPLIT_ZODI_MOON_RATIO_BOUND. 0.95 also un-reverses "
            "all 18 measured rows but costs +8.4%% chi2 against +2.8%% at 0.85."
        ),
    )
    parser.add_argument(
        "--no-halpha-centring",
        action="store_true",
        help=(
            "Keep the science-line windows at rest wavelength instead of "
            "sliding them to the per-row Halpha velocity.  Centring lifts core "
            "coverage on strong-Halpha rows from 89.0%% to 92.5%%; use this to "
            "A/B it.  Ignored when --no-science-line-mask is given."
        ),
    )
    parser.add_argument(
        "--only-thin",
        action="store_true",
        help=(
            "Skip decomposition and extract-compact steps; regenerate only the "
            "every10-thinned FITS from already-existing decomp files."
        ),
    )
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help=(
            "Write resumable META/COEF/COEF_ERR plus continuous-LSF products "
            "without full wavelength cubes. Invalid rows are retained with "
            "NaN coefficients and an explicit error status."
        ),
    )
    return parser


def _run_kwargs_from_args(args):
    """Keyword arguments `run` receives for parsed command-line `args`."""
    return dict(
        data_file=args.data_file,
        palace_dir=args.palace_dir,
        n_workers=args.n_workers,
        lsf_sigma=args.lsf_sigma,
        factor=args.factor,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_in_flight=args.max_in_flight,
        fit_model=args.fit_model,
        n_refinement_cycles=args.n_refinement_cycles,
        limit=args.limit,
        pin_workers=args.pin_workers,
        diagnose_threads=args.diagnose_threads,
        palace_suffix=args.palace_suffix,
        palace_oh_suffix=args.palace_oh_suffix,
        palace_diffuse_suffix=args.palace_diffuse_suffix,
        exposure_seconds=args.exposure_seconds,
        moon_zodi_data_root=args.moon_zodi_data_root,
        n_spline_knots=args.n_spline_knots,
        n_zodi_spline_knots=args.n_zodi_spline_knots,
        zodi_smooth_lambda=args.zodi_smooth_lambda,
        moon_smooth_lambda=args.moon_smooth_lambda,
        diffuse_ratio_bound_dex=args.diffuse_ratio_bound_dex,
        diffuse_oh_bound_dex=args.diffuse_oh_bound_dex,
        diffuse_oh_centre_log10=args.diffuse_oh_centre_log10,
        diffuse_ratio_nominal=(
            None if not args.diffuse_ratio_nominal
            else tuple(float(v) for v in args.diffuse_ratio_nominal.split(","))
        ),
        mask_science_lines=not args.no_science_line_mask,
        centre_on_halpha=not args.no_halpha_centring,
        fit_pixel_weights=not args.no_fit_pixel_weights,
        fit_pixel_weight_clip=args.fit_pixel_weight_clip,
        reversal_retry_bound=(
            None if args.no_reversal_retry else args.reversal_retry_bound
        ),
        compact_only=args.compact_only,
    )


def decompose_in_process(data_file, rows=(0,), kinds=("sci", "sky1", "sky2"),
                         argv=()):
    """Decompose rows of a stack in THIS process, exactly as a CLI run would.

    ``argv`` are extra command-line options; with none, every setting is the
    parser default, i.e. what a bare ``python -m skysub.decompose_parallel
    <data_file>`` -- a default cluster run -- uses: fit model, knots,
    smoothing, all identifiability constraints, the zodi correction
    (``LVMSKY_ZODI_CORRECTION``, read at import as for a run), photon pixel
    weights, the science-line mask centred on the measured Halpha velocity,
    the per-row telluric transmission, the geometry amplitude priors and the
    reversal retry.  The worker is initialised through the same
    `_worker_initargs` as `run`, and each row goes through the same
    `_fit_ivar_row` -> `_fit_worker_row` path as `fit_chunk_worker`.

    ``data_file`` is a stack in the corpus format (WAVE, FLUX_SCI,
    FLUX_SKY_NEAR, FLUX_SKY_FAR, LSF_SCI, LSF_SKY_NEAR, LSF_SKY_FAR and a META
    row per spectrum with the lvm_medians columns): a path, or an in-memory
    ``astropy.io.fits.HDUList`` with those HDUs, so a single exposure needs no
    file on disk.  Returns ``{(kind, row): (result, reliability_columns)}``.
    """
    in_memory = isinstance(data_file, fits.HDUList)
    args = build_arg_parser().parse_args(
        ["<in-memory stack>" if in_memory else str(data_file), *map(str, argv)])
    kw = _run_kwargs_from_args(args)
    fit_model = kw["fit_model"]
    palace_oh_suffix = _resolved_palace_oh_suffix(fit_model, kw["palace_oh_suffix"])
    base_dir, moon_zodi_root = resolve_runtime_data_roots(
        fit_model, palace_dir=kw["palace_dir"],
        moon_zodi_data_root=kw["moon_zodi_data_root"])
    wave = (np.asarray(data_file["WAVE"].data) if in_memory
            else fits.getdata(str(data_file), "WAVE")).astype(np.float64)
    init_worker(*_worker_initargs(
        wave=wave, lsf_sigma=kw["lsf_sigma"], base_dir=base_dir,
        factor=kw["factor"], data_file=data_file, progress_queue=None,
        fit_model=fit_model, n_refinement_cycles=kw["n_refinement_cycles"],
        worker_counter=None, pin_workers=False, diagnose_threads=False,
        palace_suffix=kw["palace_suffix"], palace_oh_suffix=palace_oh_suffix,
        palace_diffuse_suffix=kw["palace_diffuse_suffix"],
        exposure_seconds=kw["exposure_seconds"],
        moon_zodi_data_root=moon_zodi_root,
        n_spline_knots=kw["n_spline_knots"],
        n_zodi_spline_knots=kw["n_zodi_spline_knots"],
        zodi_smooth_lambda=kw["zodi_smooth_lambda"],
        moon_smooth_lambda=kw["moon_smooth_lambda"],
        mask_science_lines=kw["mask_science_lines"],
        centre_on_halpha=kw["centre_on_halpha"],
        fit_pixel_weights=kw["fit_pixel_weights"],
        fit_pixel_weight_clip=kw["fit_pixel_weight_clip"],
        reversal_retry_bound=kw["reversal_retry_bound"],
        diffuse_ratio_bound_dex=kw["diffuse_ratio_bound_dex"],
        diffuse_ratio_nominal=kw["diffuse_ratio_nominal"],
        diffuse_oh_centre_log10=kw["diffuse_oh_centre_log10"],
        diffuse_oh_bound_dex=kw["diffuse_oh_bound_dex"],
        compact_cache_root=None, run_fingerprint=None))
    # init_worker clamps native threads for pool workers; that is harmless here.
    out = {}
    for kind in kinds:
        for idx in rows:
            flux_row = np.asarray(_WORKER_FLUX[kind][int(idx)], dtype=np.float64) * _WORKER_FACTOR
            out[(kind, int(idx))] = _fit_worker_row(
                kind, int(idx), flux_row, _fit_ivar_row(kind, int(idx), flux_row))
    return out


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.max_in_flight is None:
        args.max_in_flight = args.n_workers
    if args.max_in_flight < 1:
        raise ValueError("--max-in-flight must be >= 1")
    if args.n_refinement_cycles < 1:
        raise ValueError("--n-refinement-cycles must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.only_thin and args.compact_only:
        raise ValueError("--only-thin and --compact-only are mutually exclusive")
    if args.compact_only and args.fit_model not in TELLURIC_FIT_MODELS:
        raise ValueError("--compact-only currently requires a telluric fit model")
    if not np.isfinite(args.exposure_seconds) or args.exposure_seconds <= 0.0:
        raise ValueError("--exposure-seconds must be positive and finite")
    if args.fit_model == MOON_ZODI_FIT_MODEL and args.exposure_seconds != 900.0:
        raise ValueError(
            "Moon/Zodi v1 records missing META exposure time as 'assumed_900s'; "
            "--exposure-seconds must therefore remain 900"
        )
    if (
        not args.only_thin
        and args.fit_model
        not in (MOON_ZODI_FIT_MODEL, *SPLIT_ZODI_FIT_MODELS, *TELLURIC_FIT_MODELS)
        and args.palace_dir is None
    ):
        parser.error(
            "palace_dir is required for baseline and non-split "
            "lsf-surface-iterative fits"
        )

    suffix = FIT_MODEL_SUFFIXES[args.fit_model]
    stem = Path(args.data_file).stem
    output_dir = Path(args.output_dir)

    if not args.only_thin:
        run(**_run_kwargs_from_args(args))

        if args.compact_only:
            with fits.open(args.data_file, memmap=True, lazy_load_hdus=True) as hdul:
                fits.HDUList(
                    [fits.PrimaryHDU(), _copy_hdu_with_name(hdul["META"], "META")]
                ).writeto(output_dir / f"{stem}_meta_only.fits", overwrite=True)
        else:
            extract_meta_and_coef_products(
                input_fits_path=args.data_file,
                decomp_fits_path_1=output_dir / f"{stem}_decomp_sky1{suffix}.fits",
                decomp_fits_path_2=output_dir / f"{stem}_decomp_sky2{suffix}.fits",
                decomp_fits_path_3=output_dir / f"{stem}_decomp_sci{suffix}.fits",
                meta_output_path=output_dir / f"{stem}_meta_only.fits",
                sky1_output_path=output_dir / f"{stem}_sky1_meta_coef{suffix}.fits",
                sky2_output_path=output_dir / f"{stem}_sky2_meta_coef{suffix}.fits",
                sci_output_path=output_dir / f"{stem}_sci_meta_coef{suffix}.fits",
            )
    else:
        required = [
            Path(args.data_file),
            output_dir / f"{stem}_decomp_sci{suffix}.fits",
            output_dir / f"{stem}_decomp_sky1{suffix}.fits",
            output_dir / f"{stem}_decomp_sky2{suffix}.fits",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "--only-thin requires these files to already exist: " + ", ".join(missing)
            )
        print(
            "--only-thin: skipping decomposition and extract-compact; "
            "regenerating thinned files only"
        )

    if not args.compact_only:
        for kind in ("sci", "sky1", "sky2"):
            thin_fits_every_n(
                output_dir / f"{stem}_decomp_{kind}{suffix}.fits",
                output_dir / f"{stem}_every10_decomp_{kind}{suffix}.fits",
                10,
            )
        thin_fits_every_n(
            args.data_file,
            output_dir / f"{stem}_every10.fits",
            10,
        )

if __name__ == "__main__":
    main()
