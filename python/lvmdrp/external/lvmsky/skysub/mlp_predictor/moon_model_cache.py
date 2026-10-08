"""Cached per-arm predictions from the frozen physical Moon/Zodi model.

Why a cache exists at all
-------------------------
``MoonZodiPhysicalModel.predict`` costs ~0.1-0.25 s per arm (the LSF projection
operator dominates; it and the ephemeris are memoised, so the second, physics-
only evaluation of an arm is cheap), i.e. ~0.55 s per row for all three arms and
both variants.  A full corpus is ~25 min on 10 cores -- far too slow to sit
inside the training loop, and an every10 subsample is sliced from its parent's
cache (`derive_from_parent`) rather than recomputed.  That is far too
slow to sit inside the training loop, but the numbers are a pure function of
(expnum, pointing, LSF) and never change once the decomposition products are
written -- so they are computed once, in parallel, and stored beside the corpus.

What it is for
--------------
The model's moon TRANSFER RATIO, ``moon(sci) / moon(near)``, predicts the
science moon amplitude better than the trained network does.  On the gaia-stars
test split (held out by night, moon-up, n = 583), error in dex:

    flat, sci = near arm       MAD 0.0228   p90 0.0747
    model, near x r_model      MAD 0.0119   p90 0.0466
    the ML network             MAD 0.0139   p90 0.0475

and the model wins in every stratum of |log10 r_model| except one holding 5
rows.  rho(log r_model, log true ratio) = +0.794, against +0.067 for the crude
``moon_signal_proxy`` the network already receives -- so this is new
INFORMATION, not a rearrangement of existing inputs, which is what separates it
from the six refuted attempts recorded in ``ablations.RETIRED``.

NOTE this reverses an earlier refutation of the same ratio.  That test used
new-oh-2 TAIL rows only and reported the opposite ordering; a 53-row version of
the test above also pointed the wrong way.  Do not re-refute it on a small or
selected subsample.

What is cached
--------------
Per row and per arm (``near``, ``far``, ``sci``): the wavelength-integrated
moon and zodi predictions in fit units, plus the geometry the model derived --
``moon_separation_deg``, ``target_airmass``, ``moon_airmass``,
``moon_altitude_deg``, ``sun_altitude_deg``, ``zodi_b500``.  Rows the model
cannot handle store NaN and ``ok = False`` rather than being dropped, so the
cache is always aligned one-to-one with the corpus rows.

**v2 additionally stores the PHYSICS-ONLY prediction** as ``zodi_total_po``
and ``moon_frac_po``.  The fields above come from the model with its FITTED
parameters; the decomposition's amplitude prior uses ``_physics_only_model``,
which zeroes the ten learned scale factors and keeps only the below-horizon
taper.  They agree on the moon FRACTION to a fraction of a percent, so the
transfer ratio below is unaffected, but they differ by 1.5-1.7x on the
ABSOLUTE zodi -- enough that comparing a fitted zodi against the v1 field
makes the Leinert anchor look unbound when 87.5% of moon-up rows are pinned
to it.  Anything that compares against a bracket the QP enforced, and the two
context features ``zodi_po_log10`` / ``moon_frac_po``, must use the v2 fields.

``upgrade_v1_to_v2`` adds them to an existing v1 cache in place, computing
only the new pass and verifying the learned fields are unchanged before it
writes, so the deployed ``moon_model_log_ratio`` comes through bit-identical
at half the cost of a rebuild.

The integrals are stored, not the spectra: a per-arm spectrum cache would be
14 500 x 3 x 12 401 floats (~2 GB) and every use so far is an amplitude ratio.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy.io import fits

CACHE_VERSION = 2
CACHE_BASENAME = "{stem}_moonzodi_model_cache_v{version}.npz"
ARMS = ("near", "far", "sci")
_ARM_META = {
    "near": ("sky_near", "sky_near_ra", "sky_near_dec", "LSF_SKY_NEAR"),
    "far": ("sky_far", "sky_far_ra", "sky_far_dec", "LSF_SKY_FAR"),
    "sci": ("sci", "sci_ra", "sci_dec", "LSF_SCI"),
}
# decompose_parallel's own default when no exposure column is available.  The
# corpus META carries `exposure`, which is an exposure NUMBER, not a duration.
# MoonZodiObservation validates the SOURCE string against a closed set, so it
# must be exactly 'assumed_900s' here -- anything else raises in __post_init__,
# and with a blanket except that shows up as a cache full of NaN.
DEFAULT_EXPOSURE_SECONDS = 900.0
EXPOSURE_SOURCE = "assumed_900s"
FIT_FLUX_SCALE = 1e14

_FIELDS = ("moon_total", "zodi_total", "moon_sep_deg", "target_airmass",
           "moon_airmass", "moon_alt_deg", "sun_alt_deg", "zodi_b500")

# v2 (2026-09-09): the PHYSICS-ONLY variant of the same two integrals.
#
# The fields above come from `MoonZodiPhysicalModel()` with its FITTED
# parameters.  The decomposition's amplitude prior does NOT use that variant:
# `geometry_amplitude_prior` goes through `_physics_only_model`, which zeroes
# all ten learned scale factors and keeps only the below-horizon taper.  The
# two agree on the moon FRACTION to a fraction of a percent -- that ratio is
# calibration-free, which is why `transfer_ratio` and the
# `moon_model_log_ratio` feature are unaffected -- but they disagree badly on
# the ABSOLUTE zodi, because the learned set contains
# `zodi_target_airmass_log = 1.28`.
#
# That mattered once already: comparing the fitted zodi against the v1
# `zodi_total` made the Leinert anchor look almost never binding (1.7% of
# moon-up rows at the ceiling, the ratio spanning 3.7 dex and 16.5% of rows
# "below the floor", which is impossible for an enforced bound).  With the
# physics-only prediction the same measurement gives 87.5%.  Anything that
# compares against a bracket the QP actually enforced must use these fields.
#
# `moon_frac_po` is `int(moon)/int(moon+zodi)`, i.e. exactly the quantity the
# moon-share bracket is stated in, so it is also the natural gate for asking
# whether a row is pinned.
_FIELDS_PO = ("zodi_total_po", "moon_frac_po")

# Worker-global state, set once per process by _init_worker.  Passing the flux
# stack through the pool would pickle gigabytes per task.
_W = {}


def cache_path(corpus_prefix, version=CACHE_VERSION):
    """Cache location: beside the corpus, so it moves with it.

    The STEM is part of the filename.  The full corpus and its every10
    subsample live in the same directory, so a stem-free name would have the
    every10 build silently overwrite the corpus cache -- and since `load`
    validates on `expnum`, the symptom would be a confusing refusal on the
    corpus rather than an obvious clobber.
    """
    prefix = Path(str(corpus_prefix))
    return prefix.parent / CACHE_BASENAME.format(stem=prefix.name,
                                                 version=version)


def _sanitised_lsf(lsf):
    """Detector LSF with unusable pixels interpolated, or None if hopeless.

    Same policy as ``decompose_parallel._sanitised_lsf_row``: the gaia1over100
    selection has 9 rows carrying a single 0.0 LSF pixel at a spectrograph arm
    join, and ``predict`` rightly refuses a non-positive FWHM.  A one-pixel
    repair cannot move a band integral measurably; a row with no usable pixel
    at all is refused rather than fitted against a fabricated LSF.
    """
    lsf = np.asarray(lsf, dtype=np.float64)
    good = np.isfinite(lsf) & (lsf > 0.0)
    if good.all():
        return lsf
    if not good.any():
        return None
    bad = np.flatnonzero(~good)
    if np.any(bad == 0) or np.any(bad == lsf.size - 1):
        return None
    if np.any(~good[bad - 1]) or np.any(~good[bad + 1]):
        return None
    idx = np.arange(lsf.size)
    out = lsf.copy()
    out[~good] = np.interp(idx[~good], idx[good], lsf[good])
    return out


def _init_worker(stack_path, exposure_seconds, zodi_correction="none"):
    from sky_decomp.moon_zodi_model import MoonZodiPhysicalModel, set_memoisation
    # Inputs are fixed for the life of a worker, so the projection operator and
    # ephemeris can be shared between the fitted and physics-only evaluations
    # of each arm and across the three arms of an exposure (~2.3x faster).
    set_memoisation(True)
    hdul = fits.open(str(stack_path), memmap=True)
    wave = np.asarray(hdul["WAVE"].data, dtype=np.float64)
    _W["hdul"] = hdul
    _W["wave"] = wave if wave.ndim == 1 else wave[0]
    _W["meta"] = hdul["META"].data
    _W["lsf"] = {a: hdul[_ARM_META[a][3]].section for a in ARMS}
    _W["model"] = MoonZodiPhysicalModel()
    _W["exposure_seconds"] = float(exposure_seconds)
    _W["zodi_correction"] = str(zodi_correction)


def _run_chunk_po(rows):
    """Physics-only integrals for one chunk (v2 fields only).

    Kept separate from `_run_chunk` so `upgrade_v1_to_v2` can add the new
    fields without recomputing the learned ones -- that keeps the existing
    `moon_model_log_ratio` feature bit-identical across the upgrade, and
    halves the cost of getting there.
    """
    from sky_decomp.moon_zodi_model import (MoonZodiObservation,
                                            geometry_amplitude_prior)
    meta, wave = _W["meta"], _W["wave"]
    out = {f"{a}_{f}": np.full(len(rows), np.nan) for a in ARMS
           for f in _FIELDS_PO}
    ok = np.zeros((len(rows), len(ARMS)), dtype=bool)
    fails = []
    for i, r in enumerate(rows):
        m = meta[int(r)]
        raw = m["date_obs"]
        date_obs = (raw.decode().strip() if isinstance(raw, bytes)
                    else str(raw).strip())
        for j, arm in enumerate(ARMS):
            role, ra_col, dec_col, _ = _ARM_META[arm]
            lsf = _sanitised_lsf(np.asarray(_W["lsf"][arm][int(r)]))
            if lsf is None:
                continue
            try:
                frac, zodi_total, _airmass = geometry_amplitude_prior(
                    wave, lsf,
                    MoonZodiObservation(
                        expnum=int(m["expnum"]), date_obs=date_obs, role=role,
                        target_ra_deg=float(m[ra_col]),
                        target_dec_deg=float(m[dec_col]),
                        exposure_seconds=_W["exposure_seconds"],
                        exposure_seconds_source=EXPOSURE_SOURCE),
                    physical_to_fit_flux_scale=FIT_FLUX_SCALE,
                    zodi_correction=_W.get("zodi_correction", "none"))
            except Exception as exc:
                # Same policy as _run_chunk: keep the first few reasons so
                # build() can raise with them instead of writing all-NaN.
                if len(fails) < 5:
                    fails.append((int(r), arm, f"{type(exc).__name__}: {exc}"))
                continue
            out[f"{arm}_zodi_total_po"][i] = float(zodi_total)
            out[f"{arm}_moon_frac_po"][i] = float(frac)
            ok[i, j] = True
    return rows, out, ok, fails


def _run_chunk(rows):
    from sky_decomp.moon_zodi_model import MoonZodiObservation
    from sky_decomp.moon_zodi_model import geometry_amplitude_prior
    meta, wave, model = _W["meta"], _W["wave"], _W["model"]
    out = {f"{a}_{f}": np.full(len(rows), np.nan) for a in ARMS
           for f in _FIELDS + _FIELDS_PO}
    ok = np.zeros((len(rows), len(ARMS)), dtype=bool)
    fails = []
    for i, r in enumerate(rows):
        m = meta[int(r)]
        raw = m["date_obs"]
        date_obs = (raw.decode().strip() if isinstance(raw, bytes)
                    else str(raw).strip())
        for j, arm in enumerate(ARMS):
            role, ra_col, dec_col, _ = _ARM_META[arm]
            lsf = _sanitised_lsf(np.asarray(_W["lsf"][arm][int(r)]))
            if lsf is None:
                continue
            try:
                pred = model.predict(
                    wave, lsf,
                    MoonZodiObservation(
                        expnum=int(m["expnum"]), date_obs=date_obs, role=role,
                        target_ra_deg=float(m[ra_col]),
                        target_dec_deg=float(m[dec_col]),
                        exposure_seconds=_W["exposure_seconds"],
                        exposure_seconds_source=EXPOSURE_SOURCE),
                    physical_to_fit_flux_scale=FIT_FLUX_SCALE)
            except Exception as exc:
                # Geometry the model refuses, or a bad row.  NaN + ok=False
                # keeps the cache aligned with the corpus.  The reason is kept
                # and surfaced by build(): a blanket except here once turned a
                # one-word argument error into a silently all-NaN cache.
                if len(fails) < 5:
                    fails.append((int(r), arm, f"{type(exc).__name__}: {exc}"))
                continue
            g = pred.state.geometry
            out[f"{arm}_moon_total"][i] = float(np.nansum(pred.moon))
            out[f"{arm}_zodi_total"][i] = float(np.nansum(pred.zodi))
            out[f"{arm}_moon_sep_deg"][i] = float(g.moon_separation_deg)
            out[f"{arm}_target_airmass"][i] = float(g.target_airmass)
            out[f"{arm}_moon_airmass"][i] = float(g.moon_airmass)
            out[f"{arm}_moon_alt_deg"][i] = float(g.moon_altitude_deg)
            out[f"{arm}_sun_alt_deg"][i] = float(g.sun_altitude_deg)
            out[f"{arm}_zodi_b500"][i] = float(g.zodi_b500)
            # Second call, through the physics-only variant: this is the
            # prediction the decomposition's amplitude prior actually used.
            try:
                _frac_po, _zodi_po, _ = geometry_amplitude_prior(
                    wave, lsf,
                    MoonZodiObservation(
                        expnum=int(m["expnum"]), date_obs=date_obs, role=role,
                        target_ra_deg=float(m[ra_col]),
                        target_dec_deg=float(m[dec_col]),
                        exposure_seconds=_W["exposure_seconds"],
                        exposure_seconds_source=EXPOSURE_SOURCE),
                    physical_to_fit_flux_scale=FIT_FLUX_SCALE,
                    zodi_correction=_W.get("zodi_correction", "none"))
            except Exception as exc:
                if len(fails) < 5:
                    fails.append((int(r), arm,
                                  f"physics-only {type(exc).__name__}: {exc}"))
                continue
            out[f"{arm}_zodi_total_po"][i] = float(_zodi_po)
            out[f"{arm}_moon_frac_po"][i] = float(_frac_po)
            ok[i, j] = True
    return rows, out, ok, fails



def decomposition_zodi_correction(corpus_prefix):
    """The Leinert zodi correction the corpus's decomposition was anchored with.

    Read from the ZODICORR primary-header keyword of ``<prefix>_decomp_*.fits``
    (written by decompose_parallel since 2026-09-24).  Products written before
    then carry no keyword and were all built uncorrected, so a missing keyword
    means ``"none"``.  Several decomposition files that DISAGREE raise, because
    then there is no single right answer for the cache to match.
    """
    import glob
    tags = {}
    for path in sorted(glob.glob(f"{corpus_prefix}_decomp_*.fits")):
        try:
            tags[path] = str(fits.getheader(path, 0).get("ZODICORR", "none")).strip()
        except OSError:
            continue
    found = set(tags.values())
    if len(found) > 1:
        raise RuntimeError(
            f"decomposition products under {corpus_prefix} disagree on the zodi "
            f"correction: " + ", ".join(f"{Path(k).name}={v}" for k, v in tags.items()))
    return found.pop() if found else "none"

def build(corpus_prefix, n_workers=8, chunk_size=32,
          exposure_seconds=DEFAULT_EXPOSURE_SECONDS, rows=None,
          overwrite=False, verbose=True, zodi_correction=None):
    """Compute the cache with a process pool and write it beside the corpus.

    ``rows`` restricts the computation (for testing); a partial cache is marked
    ``complete = False`` and ``load`` refuses it, so a smoke test can never be
    mistaken for the real thing.
    """
    import multiprocessing as mp
    import time

    stack = Path(f"{corpus_prefix}.fits")
    if not stack.exists():
        raise FileNotFoundError(f"corpus stack not found: {stack}")
    if zodi_correction is None:
        zodi_correction = decomposition_zodi_correction(corpus_prefix)
    if verbose:
        print(f"[moon-model-cache] zodi correction: {zodi_correction!r} "
              f"(from the decomposition headers)", flush=True)
    out_path = cache_path(corpus_prefix)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} exists; pass overwrite=True to rebuild it")
    with fits.open(stack, memmap=True) as hdul:
        n_rows = int(hdul["FLUX_SCI"].shape[0])
        _meta = hdul["META"].data
        expnum = np.asarray(_meta["expnum"], dtype=np.int64)
        # The science pointing is stored so a consumer holding only a triplet
        # can verify the cache belongs to ITS file.  A row-count check is not
        # enough: an every10 triplet's row_index (0..1446) fits happily inside
        # a 14 469-row corpus cache while meaning entirely different spectra,
        # which would attach every row's geometry to the wrong exposure.
        sci_ra = np.asarray(_meta["sci_ra"], dtype=np.float64)
        sci_dec = np.asarray(_meta["sci_dec"], dtype=np.float64)
    all_rows = np.arange(n_rows) if rows is None else np.asarray(rows, dtype=int)
    complete = rows is None
    chunks = [all_rows[i:i + int(chunk_size)]
              for i in range(0, all_rows.size, int(chunk_size))]
    res = {f"{a}_{f}": np.full(n_rows, np.nan) for a in ARMS
           for f in _FIELDS + _FIELDS_PO}
    ok = np.zeros((n_rows, len(ARMS)), dtype=bool)
    t0 = time.perf_counter()
    if verbose:
        print(f"[moon-model-cache] {all_rows.size} rows x {len(ARMS)} arms on "
              f"{n_workers} workers ({len(chunks)} chunks of {chunk_size})",
              flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=int(n_workers), initializer=_init_worker,
                  initargs=(str(stack), float(exposure_seconds),
                            zodi_correction)) as pool:
        done = 0
        fails = []
        for chunk_rows, vals, chunk_ok, chunk_fails in pool.imap_unordered(
                _run_chunk, chunks):
            for key, arr in vals.items():
                res[key][chunk_rows] = arr
            ok[chunk_rows] = chunk_ok
            if len(fails) < 5:
                fails.extend(chunk_fails[:5 - len(fails)])
            done += len(chunk_rows)
            if verbose and (done % (20 * int(chunk_size)) < int(chunk_size)):
                el = time.perf_counter() - t0
                print(f"  {done}/{all_rows.size}  {el:.0f}s elapsed, "
                      f"~{el / max(done, 1) * (all_rows.size - done):.0f}s left",
                      flush=True)
    payload = dict(res)
    payload["ok"] = ok
    payload["arms"] = np.array(ARMS)
    payload["fields"] = np.array(_FIELDS + _FIELDS_PO)
    payload["expnum"] = expnum
    payload["sci_ra"] = sci_ra
    payload["sci_dec"] = sci_dec
    payload["n_rows"] = np.array(n_rows)
    payload["complete"] = np.array(bool(complete))
    payload["version"] = np.array(CACHE_VERSION)
    # Which zodi correction the physics-only fields were computed under; see
    # decomposition_zodi_correction and load().
    payload["zodi_correction"] = np.array(str(zodi_correction))
    payload["exposure_seconds"] = np.array(float(exposure_seconds))
    payload["fit_flux_scale"] = np.array(FIT_FLUX_SCALE)
    np.savez_compressed(out_path, **payload)
    # Report the usable fraction over the rows actually COMPUTED, not over the
    # whole corpus -- otherwise a `rows=` smoke test always looks like a
    # near-total failure.
    usable = ok[all_rows].all(axis=1)
    if not usable.any():
        raise RuntimeError(
            "the moon-model cache came out empty: no row produced all three "
            "arms. First failures: "
            + "; ".join(f"row {r} {a}: {m}" for r, a, m in fails[:5]))
    if verbose:
        print(f"[moon-model-cache] wrote {out_path} in "
              f"{time.perf_counter() - t0:.0f}s; all three arms usable on "
              f"{usable.mean() * 100.0:.2f}% of the {all_rows.size} rows "
              f"computed"
              + ("" if complete else "  (PARTIAL -- load() will refuse it)"))
        if fails:
            print("  first failures: "
                  + "; ".join(f"row {r} {a}: {m}" for r, a, m in fails[:3]))
    return out_path


def upgrade_v1_to_v2(corpus_prefix, n_workers=8, chunk_size=32, verbose=True):
    """Add the v2 physics-only fields to an existing v1 cache, in place.

    Only the new fields are computed, so the learned-parameter fields -- and
    therefore `transfer_ratio` and the deployed `moon_model_log_ratio` feature
    -- come through bit-identical.  That is verified before the write, not
    assumed.  Halves the cost against a full rebuild, which would have to
    evaluate both model variants for every row.

    The v1 file is left on disk under its own name (the version is part of the
    filename), so this is additive and reversible.
    """
    import multiprocessing as mp
    import time

    src = cache_path(corpus_prefix, version=1)
    dst = cache_path(corpus_prefix, version=2)
    if not src.exists():
        raise FileNotFoundError(f"no v1 cache to upgrade at {src}")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; delete it to redo the upgrade")
    stack = Path(f"{corpus_prefix}.fits")
    if not stack.exists():
        raise FileNotFoundError(f"corpus stack not found: {stack}")
    old = {k: v for k, v in np.load(src, allow_pickle=False).items()}
    if int(old["version"]) != 1:
        raise RuntimeError(f"{src} is version {int(old['version'])}, not 1")
    if not bool(old["complete"]):
        raise RuntimeError(f"{src} is a PARTIAL cache; rebuild rather than upgrade")
    n_rows = int(old["n_rows"])
    with fits.open(stack, memmap=True) as hdul:
        if int(hdul["FLUX_SCI"].shape[0]) != n_rows:
            raise RuntimeError(
                f"{src} describes {n_rows} rows but {stack} has "
                f"{int(hdul['FLUX_SCI'].shape[0])}; wrong prefix")
        if not np.array_equal(np.asarray(hdul["META"].data["expnum"], np.int64),
                              np.asarray(old["expnum"], np.int64)):
            raise RuntimeError(f"{src} does not describe {stack} (expnum differs)")
    exposure_seconds = float(old["exposure_seconds"])
    rows_all = np.arange(n_rows)
    chunks = [rows_all[i:i + int(chunk_size)]
              for i in range(0, n_rows, int(chunk_size))]
    res = {f"{a}_{f}": np.full(n_rows, np.nan) for a in ARMS for f in _FIELDS_PO}
    ok_po = np.zeros((n_rows, len(ARMS)), dtype=bool)
    t0 = time.perf_counter()
    if verbose:
        print(f"[moon-model-cache] upgrading {src.name} -> v2: {n_rows} rows x "
              f"{len(ARMS)} arms of physics-only prediction on {n_workers} "
              f"workers ({len(chunks)} chunks of {chunk_size})", flush=True)
    ctx = mp.get_context("fork")
    fails = []
    with ctx.Pool(processes=int(n_workers), initializer=_init_worker,
                  initargs=(str(stack), exposure_seconds,
                            decomposition_zodi_correction(corpus_prefix))) as pool:
        done = 0
        for chunk_rows, vals, chunk_ok, chunk_fails in pool.imap_unordered(
                _run_chunk_po, chunks):
            for key, arr in vals.items():
                res[key][chunk_rows] = arr
            ok_po[chunk_rows] = chunk_ok
            if len(fails) < 5:
                fails.extend(chunk_fails[:5 - len(fails)])
            done += len(chunk_rows)
            if verbose and (done % (20 * int(chunk_size)) < int(chunk_size)):
                el = time.perf_counter() - t0
                print(f"  {done}/{n_rows}  {el:.0f}s elapsed, "
                      f"~{el / max(done, 1) * (n_rows - done):.0f}s left",
                      flush=True)
    usable = ok_po.all(axis=1)
    if not usable.any():
        raise RuntimeError(
            "the physics-only pass produced nothing. First failures: "
            + "; ".join(f"row {r} {a}: {m}" for r, a, m in fails[:5]))
    payload = dict(old)
    payload.update(res)
    # `ok` becomes the AND of both passes: a row is usable only if both model
    # variants produced all three arms, so downstream gates stay one flag.
    payload["ok"] = np.asarray(old["ok"], dtype=bool) & ok_po
    payload["fields"] = np.array(_FIELDS + _FIELDS_PO)
    payload["version"] = np.array(2)
    payload["zodi_correction"] = np.array(decomposition_zodi_correction(corpus_prefix))
    # The learned fields must be untouched -- checked, not assumed.
    for a in ARMS:
        for f in _FIELDS:
            k = f"{a}_{f}"
            if not np.array_equal(payload[k], old[k], equal_nan=True):
                raise RuntimeError(f"upgrade would have changed {k}; refusing")
    np.savez_compressed(dst, **payload)
    if verbose:
        _drop = int((np.asarray(old['ok'], bool).all(axis=1) & ~usable).sum())
        print(f"[moon-model-cache] wrote {dst} in "
              f"{time.perf_counter() - t0:.0f}s; physics-only usable on "
              f"{usable.mean() * 100.0:.2f}% of {n_rows} rows"
              + (f"; {_drop} row(s) usable in v1 are not usable in the "
                 f"physics-only pass" if _drop else ""))
        if fails:
            print("  first failures: "
                  + "; ".join(f"row {r} {a}: {m}" for r, a, m in fails[:3]))
    return dst


def load(corpus_prefix, expnum=None, require_complete=True):
    """Load the cache, validating it against the corpus it claims to describe.

    ``expnum`` (the corpus META column) is checked element-wise.  That is the
    self-invalidation: a cache built for a different selection, or for a corpus
    whose rows were reordered, is refused rather than silently misaligned --
    which would attach every row's geometry to the wrong spectrum.
    """
    path = cache_path(corpus_prefix)
    if not path.exists():
        raise FileNotFoundError(
            f"no moon-model cache at {path}; build it with "
            f"`python -m mlp_predictor.moon_model_cache {corpus_prefix}`")
    z = np.load(path, allow_pickle=False)
    if int(z["version"]) != CACHE_VERSION:
        _hint = ("upgrade it in place with "
                 f"`python -m mlp_predictor.moon_model_cache {corpus_prefix} "
                 f"--upgrade` (~26 min, keeps the learned fields untouched)"
                 if int(z["version"]) == 1 else "rebuild it")
        raise RuntimeError(
            f"{path} is cache version {int(z['version'])}, expected "
            f"{CACHE_VERSION}; {_hint}")
    if require_complete and not bool(z["complete"]):
        raise RuntimeError(
            f"{path} is a PARTIAL cache (built with rows=...); rebuild it "
            f"without `rows` before using it for training")
    # The physics-only fields must be computed under the same zodi correction
    # the decomposition's anchor used.  The ML's zodi-ceiling rule fits ONE
    # scale S with pinned zodi = S x zodi_po; if the two disagree that ratio is
    # no longer constant and the rule is silently wrong on every pinned row.
    # Caches written before 2026-09-24 carry no tag and were all uncorrected.
    _have_zc = str(z["zodi_correction"]) if "zodi_correction" in z.files else "none"
    _want_zc = decomposition_zodi_correction(corpus_prefix)
    if _have_zc != _want_zc:
        raise RuntimeError(
            f"{path} was built with zodi correction {_have_zc!r} but the "
            f"decomposition under {corpus_prefix} was anchored with "
            f"{_want_zc!r}. Rebuild the cache (delete it and rerun, or "
            f"`python -m mlp_predictor.moon_model_cache {corpus_prefix} --overwrite`).")
    if expnum is not None:
        want = np.asarray(expnum, dtype=np.int64)
        have = np.asarray(z["expnum"], dtype=np.int64)
        if want.shape != have.shape or not np.array_equal(want, have):
            raise RuntimeError(
                f"{path} does not match this corpus: cache has {have.size} "
                f"rows, corpus has {want.size}"
                + ("" if want.shape != have.shape else
                   f", and {int((want != have).sum())} expnum values differ"))
    return {k: z[k] for k in z.files}


def default_workers():
    """Worker count for an implicit build: every core.

    Measured on an M1 Max (8 performance + 2 efficiency cores): 10 workers
    give 12.7 rows/s against 11.0 for 8, so the efficiency cores still help.
    """
    return max(1, os.cpu_count() or 1)


def _parent_prefix(corpus_prefix):
    """The full-corpus prefix an ``*_every<N>`` subsample was thinned from."""
    import re
    prefix = str(corpus_prefix)
    parent = re.sub(r"_every\d+$", "", prefix)
    return None if parent == prefix else parent


def derive_from_parent(corpus_prefix, parent_prefix=None, n_check=4,
                       rtol=1e-12, verbose=True):
    """Write a subsample's cache by slicing its parent corpus's cache.

    An every10 stack is a row subset of the full stack -- same exposures, same
    pointings, same LSF rows -- so its cache entries are exactly the parent's
    at the matching rows, and recomputing them costs ~3 minutes of CPU for
    nothing.  Rows are matched on ``expnum`` (unique per row) and the science
    pointing is checked too.  The parent must load cleanly and carry the same
    zodi correction the subsample's decomposition was anchored with, and
    ``n_check`` rows spread over the subsample are RECOMPUTED from its own
    stack and must agree to ``rtol``; any disagreement raises, so a subsample
    whose inputs differ from its parent's can never inherit the wrong numbers.
    """
    parent_prefix = parent_prefix or _parent_prefix(corpus_prefix)
    if parent_prefix is None:
        raise ValueError(f"{corpus_prefix} has no parent corpus prefix")
    parent = load(parent_prefix)
    want_zc = decomposition_zodi_correction(corpus_prefix)
    # Untagged caches predate the tag and were all uncorrected (as in load()).
    have_zc = str(parent.get("zodi_correction", "none"))
    if have_zc != want_zc:
        raise RuntimeError(
            f"parent cache zodi correction {have_zc!r} != {want_zc!r} for "
            f"{corpus_prefix}")
    stack = Path(f"{corpus_prefix}.fits")
    with fits.open(stack, memmap=True) as hdul:
        _meta = hdul["META"].data
        expnum = np.asarray(_meta["expnum"], dtype=np.int64)
        sci_ra = np.asarray(_meta["sci_ra"], dtype=np.float64)
        sci_dec = np.asarray(_meta["sci_dec"], dtype=np.float64)
    p_expnum = np.asarray(parent["expnum"], dtype=np.int64)
    order = np.argsort(p_expnum)
    pos = np.searchsorted(p_expnum[order], expnum)
    pos = np.clip(pos, 0, p_expnum.size - 1)
    idx = order[pos]
    if not np.array_equal(p_expnum[idx], expnum):
        raise RuntimeError(f"{corpus_prefix} has exposures its parent lacks")
    if not (np.array_equal(np.asarray(parent["sci_ra"])[idx], sci_ra)
            and np.array_equal(np.asarray(parent["sci_dec"])[idx], sci_dec)):
        raise RuntimeError(f"{corpus_prefix} pointings differ from its parent's")
    n_parent = int(parent["n_rows"])
    payload = {k: (np.asarray(v)[idx]
                   if np.ndim(v) >= 1 and np.shape(v)[0] == n_parent else v)
               for k, v in parent.items()}
    payload["expnum"] = expnum
    payload["sci_ra"] = sci_ra
    payload["sci_dec"] = sci_dec
    payload["n_rows"] = np.array(expnum.size)
    payload["zodi_correction"] = np.array(want_zc)

    # Spot check: recompute a few rows from the SUBSAMPLE's own stack.
    check = np.unique(np.linspace(0, expnum.size - 1, max(int(n_check), 1)).astype(int))
    from sky_decomp import moon_zodi_model as _mzm
    _was_memoising = _mzm._MEMOISE
    _init_worker(str(stack), float(parent["exposure_seconds"]), want_zc)
    try:
        _, vals, ok, _ = _run_chunk(check)
    finally:
        _W["hdul"].close()
        _W.clear()
        _mzm.set_memoisation(_was_memoising)   # _init_worker turned it on
    if not np.array_equal(ok, np.asarray(payload["ok"])[check]):
        raise RuntimeError("spot check: usable-arm flags differ from the parent's")
    for key, arr in vals.items():
        ref = np.asarray(payload[key])[check]
        fin = np.isfinite(arr)
        if not np.array_equal(fin, np.isfinite(ref)) or not np.allclose(
                arr[fin], ref[fin], rtol=rtol, atol=0.0):
            raise RuntimeError(f"spot check: {key} differs from the parent's")
    out_path = cache_path(corpus_prefix)
    np.savez_compressed(out_path, **payload)
    if verbose:
        print(f"[moon-model-cache] wrote {out_path} by slicing "
              f"{cache_path(parent_prefix).name} ({expnum.size} rows, "
              f"{check.size} rows re-verified)")
    return out_path


def load_or_build(corpus_prefix, n_workers=None, expnum=None, verbose=True,
                  chunk_size=32):
    """Load the cache, building it first if it does not exist yet.

    Mirrors the wavelength cache's contract: a corpus without a cache simply
    grows one on first use.  ABSENCE triggers a build; a cache that exists but
    does not VALIDATE does not -- it raises instead.  That asymmetry is
    deliberate.  A validation failure means either a stale cache or a caller
    that passed the wrong prefix, and silently spending ~26 minutes of CPU to
    find out which is not a reasonable default; the message says what to run.
    """
    path = cache_path(corpus_prefix)
    parent = _parent_prefix(corpus_prefix)
    if not path.exists() and parent is not None and cache_path(parent).exists():
        # A subsample of a corpus that already has its cache: slice it.
        try:
            derive_from_parent(corpus_prefix, parent, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [moon-model-cache] could not derive from {parent} "
                      f"({type(exc).__name__}: {exc}); building instead.")
    if not path.exists():
        workers = default_workers() if n_workers is None else int(n_workers)
        if verbose:
            print(f"  [moon-model-cache] no cache at {path}; building it now on "
                  f"{workers} worker(s).  This is a one-off: ~25 min for a "
                  f"full corpus on 10 cores; an every10 subsample is sliced "
                  f"from its parent's cache in seconds once that exists.")
        build(corpus_prefix, n_workers=workers, chunk_size=chunk_size,
              overwrite=False, verbose=verbose)
    return load(corpus_prefix, expnum=expnum)


def transfer_ratio(cache, reference="near", eps=1e-30):
    """``log10(moon_sci / moon_reference)`` -- the quantity that beat the network.

    NaN where either arm is unusable or non-positive, which includes every
    moon-down row: the model's moon there is ~1e-2 in fit units against ~1e4
    with the moon up, so the ratio is numerically meaningless and the consumer
    must gate on ``moon_alt_deg`` rather than trusting a finite value.
    """
    num = np.asarray(cache["sci_moon_total"], dtype=np.float64)
    den = np.asarray(cache[f"{reference}_moon_total"], dtype=np.float64)
    good = np.isfinite(num) & np.isfinite(den) & (num > eps) & (den > eps)
    out = np.full(num.shape, np.nan)
    out[good] = np.log10(num[good] / den[good])
    return out


def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Build the physical Moon/Zodi model cache for one corpus.")
    p.add_argument("corpus_prefix",
                   help="e.g. gaia-stars/lvmsframe_median_stack_1.2.1_gaia1over100 "
                        "(without the .fits)")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--rows", type=int, default=None,
                   help="only the first N rows, for a smoke test; the result is "
                        "marked PARTIAL and load() refuses it")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--upgrade", action="store_true",
                   help="add the v2 physics-only fields to an existing v1 "
                        "cache instead of rebuilding from scratch; the "
                        "learned-parameter fields are copied verbatim")
    a = p.parse_args(argv)
    if a.upgrade:
        upgrade_v1_to_v2(a.corpus_prefix, n_workers=a.n_workers,
                         chunk_size=a.chunk_size)
        return
    build(a.corpus_prefix, n_workers=a.n_workers, chunk_size=a.chunk_size,
          rows=(None if a.rows is None else np.arange(a.rows)),
          overwrite=a.overwrite)


if __name__ == "__main__":
    _main()
