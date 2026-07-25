# Barycentric Correction: Design Discussion and Implementation

Status: **implemented** on branch `barycentric-correction` (based on
`master-perf`). Sections below are kept in their original order as a design
history; see "Implementation (completed)" near the end for exactly what was
built, tested, and where it stands.

## Background: where wavelengths enter the pipeline

For context, wavelength calibration happens in stages, all before flux
calibration and channel joining (see `main.py` `science_reduction()`,
roughly lines 1673-1735):

1. **Derive solution from arcs** — `determine_wavelength_solution()`
   (`rssMethod.py:235-670`). Fits a per-fiber polynomial (pixel → wavelength)
   from arc-lamp lines. Extraction itself (`extract_spectra()`,
   `imageMethod.py:2172`) stays in pixel space: one flux value per raw CCD
   pixel column per fiber, no wavelength attached, no sub-pixel oversampling.
2. **Apply solution to science data** — `create_pixel_table()`
   (`rssMethod.py:847-901`). Evaluates the per-fiber polynomial into a full
   2D wavelength array matching the flux array
   (`RSS.set_wave_array()`, `rss.py:935-983`).
3. **Refine via sky lines** — `shift_wave_skylines()` (`rssMethod.py:712-810`).
   Adjusts the per-fiber wavelength solution using measured sky-line
   centroids (real, topocentric-frame emission lines).
4. **Resample onto common grid** — `resample_wavelength()`
   (`rssMethod.py:1151-1220`) via `RSS.rectify_wave()`
   (`rss.py:1622-1785`). Interpolates each fiber's native, non-uniform,
   per-pixel wavelength array onto one shared ~0.5 Å grid
   (`wave_range`, `wave_disp`), the same grid for every exposure.

Everything downstream of step 4 (flux calibration, sky subtraction, channel
joining) operates on that shared, fixed 0.5 Å grid.

## The goal

Apply a barycentric (or more generally, per-exposure) velocity correction
**without changing the wavelength array**. Instead, the correction should be
absorbed into the flux values, so the final wavelength grid stays identical
across all exposures.

## What already exists (and why it's not sufficient)

`RSS.get_helio_rv()` (`rss.py:3214-3284`), called from `create_pixel_table()`
(`rssMethod.py:895`), already computes heliocentric RV corrections via
`astropy.coordinates.SkyCoord.radial_velocity_correction()`:

- Separately for the **SCI, SKYE, and SKYW telescopes** (`rss.py:3235-3249`),
  since each points at a different RA/Dec.
- Separately for each **acquired standard star** (`rss.py:3252-3270`).
- All values are written to header keywords
  (`HIERARCH WAVE HELIORV_{tel}`, `STD{istd}HRV`) for provenance, but nothing
  currently *applies* them — the `apply_hrv_corr` branch is a dead stub
  (`rss.py:3272`, just `...`), with commented-out legacy code (`rss.py:3282`)
  showing the originally intended (and rejected) approach:

  ```python
  # rss._wave = rss._wave * (1 + helio_vel / c.to("km/s").value)
  ```

  i.e. directly rescaling the wavelength array — exactly what we want to
  avoid.

## Key design tension: sky subtraction vs. cross-epoch combination

Sky subtraction is `SCI_flux(λ) − SKY_flux(λ)` at matching wavelength. Sky
emission lines are physically fixed in the **topocentric** (observatory)
frame — they come from Earth's atmosphere, not the target — so they land at
the same observed wavelength in SCI and SKY fibers regardless of pointing.

- **Per-telescope correction** (what `get_helio_rv` currently computes) would
  apply a *different* shift to SCI vs. SKYE/SKYW, since they point at
  different sky coordinates. That misaligns sky emission lines between SCI
  and SKY fiber sets and degrades subtraction.
- **A single, uniform correction** applied identically to every fiber in an
  exposure (SCI, SKYE, SKYW, standards) leaves the *relative* alignment
  between SCI and SKY completely unchanged — they all shift together — so
  sky subtraction is unaffected. What it does change is the exposure's
  absolute wavelength placement, which is exactly what's needed to remove
  the season-to-season Earth-velocity modulation (up to ±30 km/s) between
  exposures taken months apart.

**Conclusion:** use one scalar RV per exposure (e.g. computed from the SCI
field-center RA/Dec), applied uniformly to all fibers — not the per-telescope
dict `get_helio_rv` currently produces. The minor cost is that SKYE/SKYW's
own true RV (relative to their actual pointing) is technically not what gets
applied, but that differential is small and irrelevant for sky fibers, whose
lines shouldn't be barycentric-corrected in the first place.

## Interpolation-error considerations

- Extraction (`extract_spectra`, `imageMethod.py:2172`) yields one flux value
  per native CCD pixel — not a fine sub-pixel grid.
- `rectify_wave()` (`rss.py:1732-1762`) reconciles each fiber's
  slightly-different native wavelength array with the shared grid using
  **point interpolation** (`scipy.interpolate.interp1d`, linear or cubic
  spline) — not flux-conserving rebinning (integration over output bin
  edges). This already happens, unconditionally, once per exposure.
- Native pixel dispersion is comparable to the ~0.5 Å output grid spacing,
  so this interpolation is a genuine resample near native resolution, not a
  safe fine→coarse binning. It already introduces some noise correlation
  between adjacent output pixels and can mildly blur narrow features
  (e.g. sky lines) — a pre-existing characteristic of the pipeline,
  independent of barycentric correction.

### Why the correction must be applied *during* the original resample, not after

±30 km/s corresponds to `Δλ/λ = v/c ≈ 1×10⁻⁴`:

- ~0.4 Å at the blue end (~3600 Å)
- ~1 Å at the red end of z-band (~9800 Å) — about **2 pixels** on the 0.5 Å
  common grid.

That is not a sub-pixel nudge. If the correction were applied **after**
exposures are already on the shared 0.5 Å grid (i.e. downstream, post
`rectify_wave`), it would require a *second*, independent interpolation of
data whose effective resolution is already capped at the grid spacing — a
second lossy pass, shifting by an amount that's large relative to that
already-degraded resolution.

If instead the correction is folded into the **original** `rectify_wave`
call — before the native per-pixel data collapses onto the common grid — it
comes essentially for free: it's the same single interpolation pass that
must happen anyway (native per-fiber wavelength array → common grid), just
with the input abscissa pre-scaled by `(1 + v/c)` before interpolating. No
second pass, and it operates on data that still has full native resolution.

This also fixes the specific downstream problem that motivated this
discussion: needing to combine exposures with barycentric corrections
differing by up to ±30 km/s, where the correction would otherwise have to be
applied to data that has already been gridded — which is exactly the lossy,
second-pass scenario above.

## Code that would need to change

1. **`RSS.get_helio_rv()`** (`rss.py:3214-3284`)
   - Keep computing header-only per-telescope/per-standard RVs for
     provenance if desired, but this is *not* the value to use for the
     applied correction.
   - Remove or replace the dead `apply_hrv_corr` stub (`rss.py:3272-3282`) —
     it currently encodes the rejected wavelength-mutation approach.
   - Likely add a method (or reuse existing logic) to compute a single
     scalar RV from the SCI field-center RA/Dec, to be threaded through to
     the resample step.

2. **`create_pixel_table()`** (`rssMethod.py:847-901`)
   - Currently calls `get_helio_rv()` and writes header keywords
     (`rssMethod.py:894-896`). Would need to also surface/return the single
     scalar RV chosen for application (or leave that entirely to the
     resample step, and just keep this function responsible for
     provenance headers).

3. **`RSS.rectify_wave()`** (`rss.py:1622-1785`) — **this is the actual
   application point**. Around `rss.py:1732-1762`, each fiber's flux,
   error, sky, and LSF arrays are interpolated via
   `interpolate.interp1d(rss._wave[ifiber][sel], rss._data[ifiber][sel], ...)`
   onto the output `wave` grid. The change: scale the *input* abscissa
   (`rss._wave[ifiber] * (1 + v/c)`) before building each interpolator,
   while leaving `wave` (the query grid, and thus `new_rss._wave`)
   untouched. Needs a new parameter (e.g. `rv_corr` or `bary_rv`, km/s,
   default 0/None) plumbed through.

4. **`resample_wavelength()`** (`rssMethod.py:1151-1220`) — needs a new
   parameter to accept the scalar RV and pass it to `rectify_wave()`.

5. **`main.py`** orchestration (`science_reduction()`, ~lines 1673-1735)
   - `create_pixel_table()` call at `main.py:1681`
   - `shift_wave_skylines()` at `main.py:1689` — this refines the
     wavelength solution against *real, topocentric* sky lines, so it
     should continue to run **before** the barycentric correction is
     applied (current ordering already supports this, since the
     correction would be injected only at the `resample_wavelength` step).
   - `resample_wavelength()` call at `main.py:1701` — needs the RV value
     threaded in from wherever it's computed (either recomputed here from
     header keywords written by `create_pixel_table`, or passed through
     the pipeline state).

## Impact on flux calibration (standard star sensitivity curves)

If a uniform per-exposure correction is applied to *all* fibers (per the
conclusion above), does it disturb the sensitivity curves derived from
standard-star fibers? Short answer: the self-consistency argument mostly
carries over, but there is one concrete, quantified risk worth respecting.

### Self-consistency still holds for the applied-calibration step

`apply_fluxcal()` (`fluxCalMethod.py:56-115`) loads one exposure and applies
its own sensitivity curve to its own flux at matching wavelength-grid index
(same `rss._wave` for STD, SCI, and SKY fibers). If every fiber in that
exposure carries the *same* uniform shift, the STD-derived sensitivity curve
and the SCI flux it's multiplied into stay mutually registered — the same
reasoning as for sky subtraction. **Standard star fibers should get the same
uniform correction as everything else, not be excluded** — excluding them
would instead break this registration.

### Where an external, un-shifted reference is involved

Two sensitivity methods exist; `apply_fluxcal` defaults to `method='MOD'`
(`fluxCalMethod.py:56`):

- **`STD`** (fallback): `standard_sensitivity()` (`fluxCalMethod.py:1655-1765`)
  divides a Gaia XP reference spectrum (external, sampled at literal nominal
  wavelength, `fluxCalMethod.py:1738`) by the observed standard spectrum, then
  heavily smooths the ratio (`make_smoothing_spline(..., lam=1e4)`,
  `fluxCalMethod.py:1741`). Gaia XP resolution is very coarse (resolution
  elements of tens of Å), and the smoothing further suppresses narrow-line
  residuals, so this path is not meaningfully sensitive to an Å-scale shift.
- **`MOD`** (**the active default** — corrected from an earlier, wrong read of
  this code that assumed the model-based path was dead): computed inline
  inside `model_selection()` (`fluxCalMethod.py:842-1313`), not via the
  separate, genuinely-unused `calc_sensitivity_from_model()` helper. The
  actual ratio (`fluxCalMethod.py:1226`):
  ```python
  sens_tmp = stack_stellar_model[i][n_chan] * model_to_gaia_median[i] / std_masked
  ```
  divides a **high-resolution AMBRE stellar template** (LVM's own resolution,
  not Gaia's) by the observed, telluric-corrected standard spectrum. This is
  a genuine line-profile-level comparison, not a coarse/smoothed one.

**Telluric correction is also active** (contrary to an earlier assumption),
using the same high-resolution machinery:
- Inside `model_selection()`: `calc_pwv()` (`fluxCalMethod.py:714`, called at
  `fluxCalMethod.py:1046`) fits precipitable water vapor per standard star via
  `TelluricCalculator`, and the resulting transmission spectrum telluric-
  corrects the standard spectrum (`fluxCalMethod.py:1216`) before the
  sensitivity ratio above is computed.
- Inside `apply_fluxcal()` (`fluxCalMethod.py:171-188`): re-derives telluric
  transmission from the header `PWV_MED` value and divides the final mean
  sensitivity curve by it, separately per SCI/SKYE/SKYW airmass.

### Why this is self-correcting in principle — and the quantified caveat

Before computing `sens_tmp`, `model_selection()` fits and removes a velocity
shift between the observed standard spectrum and the model template via
cross-correlation:

```python
# fluxCalMethod.py:975
log_shift_full = fluxcal.derive_vecshift(flux_std_logscale[mask_good], flux_model_logscale[mask_good], max_ampl=3) * ...
vel_shift_full = log_shift_full * 299792.458   # fluxCalMethod.py:977
```

Since this fit is redone per exposure per star, it should automatically
absorb whatever extra shift our correction introduces — *provided the search
window is wide enough to cover it*. `derive_vecshift()`
(`core/fluxcal.py:537-556`) caps the correlation search at `max_ampl=3`
samples of the log-wavelength grid used for matching (oversampled 20× only
for sub-sample precision within that cap, not to extend its range).

That log grid's step is the *minimum* `d(ln λ)` across the concatenated b/r/z
standard-star spectrum (`linear_to_logscale()`, `fluxCalMethod.py:327-335`),
which occurs at the reddest sampled wavelength (~9800 Å, z-band edge) given a
uniform 0.5 Å linear grid:

```
dlnλ_min  = 0.5 Å / 9800 Å           ≈ 5.10×10⁻⁵
v/pixel   = c × dlnλ_min             ≈ 15.3 km/s
capture   = max_ampl(3) × v/pixel    ≈ ±45.9 km/s  (≈92 km/s peak-to-peak)
```

So the model-matching fit can only chase a combined shift (star's true radial
velocity **+** our injected barycentric-style correction) of up to roughly
**±46 km/s** before clipping at the edge of its search window and returning a
biased or wrong velocity shift — which would then feed a wrongly-shifted
template into the sensitivity ratio and the PWV fit for that star/exposure.
Our correction alone (±30 km/s) already consumes a large fraction of that
budget; combined with a standard star of even modest RV (a few tens of
km/s in the same direction), some stars/epochs could realistically exceed the
capture range. This is a concrete, numerically-grounded risk, not a
theoretical one, and worth checking empirically (e.g. against the RV
distribution of the actual standard-star catalog in use) before relying on
this self-correction to fully absorb the injected shift.

## Implementation (completed)

Branch: `barycentric-correction`, based on `master-perf`'s tip (chosen over
branching from `upgrade` so this work carries none of the unrelated,
still-pending Python 3.12/NumPy 2 upgrade content — see the branch-strategy
discussion earlier in this project; `git checkout -b barycentric-correction
master-perf` from a worktree that already had `master-perf` checked out
elsewhere).

### Sign convention — verified numerically, not assumed

Before writing any code, the sign of the correction was pinned down with a
standalone numerical check (synthetic Gaussian line, known injected
`rv_corr`, `astropy`'s actual `radial_velocity_correction()` convention):
relabelling the native/observed wavelength array as
`wave_native = rss._wave * (1 + rv_corr / c)` and interpolating onto the
*unchanged* output grid correctly recovers the line at its true wavelength.
This matches the sign already used in the old, dead, rejected code at
`rss.py` (`rss._wave = rss._wave * (1 + helio_vel / c)`) — that old code had
the right formula, it was just applying it to the wrong target (the
wavelength array instead of the flux-interpolation abscissa).

### Files changed

- **`python/lvmdrp/core/rss.py`**
  - `RSS.rectify_wave()`: new `rv_corr: float = 0.0` (km/s) parameter. Before
    the per-fiber interpolation loop, computes
    `wave_native = rss._wave * (1 + rv_corr/c)` (`c` via
    `astropy.constants`, newly imported) and uses `wave_native` as the
    abscissa for every interpolation (`_data`, `_error`, `_mask`, `_lsf`,
    `_sky`, `_sky_error`, `_sky_east(_error)`, `_sky_west(_error)`) instead
    of `rss._wave`. The output query array `wave` — and therefore
    `new_rss._wave` — is untouched. Writes `HIERARCH WAVE RVCORR_APPLIED`
    when `rv_corr != 0`. Warns (doesn't silently drop) if `rv_corr` is
    requested on an RSS that's already rectified/1D (early-return path).
    Left an inline comment flagging the `derive_vecshift(max_ampl=3)`
    capture-range caveat (see below) for future readers.
  - New `RSS.get_bary_rv()`: computes the single scalar RV from `SCIRA` /
    `SCIDEC` + `OBSTIME`, reusing the existing `EarthLocation` pattern.
    Returns `0.0` with a warning on missing/invalid header info (mirrors
    `get_helio_rv`'s defensive style).
  - `RSS.get_helio_rv()`: dropped the dead `apply_hrv_corr` parameter and
    the commented-out stub that used to (incorrectly) mutate `rss._wave`
    directly. It still computes and writes the per-telescope
    (`HIERARCH WAVE HELIORV_{tel}`) and per-standard (`STD{istd}HRV`)
    values, now explicitly documented as informational/QA-only, distinct
    from the single value that's actually applied.
- **`python/lvmdrp/functions/rssMethod.py`**
  - `create_pixel_table()`: dropped the now-meaningless `apply_heliorv`
    parameter. Still calls `get_helio_rv()` for QA headers; additionally
    calls `get_bary_rv()` and writes `HIERARCH WAVE BARYRV_APPLIED`.
  - `resample_wavelength()`: new `apply_bary_corr: bool = True` parameter.
    Reads `HIERARCH WAVE BARYRV_APPLIED` from the header (0.0 if
    `apply_bary_corr` is False or the key is absent) and passes it to
    `rectify_wave(..., rv_corr=...)`.
- **`python/lvmdrp/main.py`**: `apply_bary_corr: bool = True` added to
  `science_reduction()` (threaded into its `resample_wavelength()` call) and
  to `run_drp()` (threaded into both the multi-MJD recursive call and the
  `science_reduction()` call).
- **`bin/drp`**: new `--apply-bary-corr/--no-apply-bary-corr` CLI flag
  (default on), threaded through to `run_drp()`, mirroring the existing
  `--fluxcal-method` flag's plumbing style.
- **`CHANGELOG.rst`**: unreleased-section entry added.
- **`tests/core/test_rss_rectify.py`** (new, no prior coverage existed for
  `rectify_wave`/`resample_wavelength`/`create_pixel_table`/`get_helio_rv`):
  synthetic-RSS fixtures, no file I/O, modeled on `tests/core/test_sky.py`.
  - Regression test: `rv_corr=0.0` (explicit or default) reproduces
    identical output.
  - Shift-correctness test: injects a known velocity into a synthetic
    Gaussian line, confirms `rectify_wave(rv_corr=v)` recovers the line at
    its true wavelength on the *unchanged* output grid, and that no
    correction leaves it at the (wrong) shifted position — this is what
    actually locks in the sign convention as a regression guard.
  - `get_bary_rv()` tests: plausible return value with a full header, and
    `0.0` fallback (no exception) for each of `SCIRA`/`SCIDEC`/`OBSTIME`/
    `IMAGETYP` missing individually.

### Header provenance — how the correction is documented in output files

Two distinct keywords, written at two different pipeline stages, together
form an audit trail:

- **`HIERARCH WAVE BARYRV_APPLIED`** — written by `create_pixel_table()`
  (early wavelength-calibration stage, e.g. the `wsci`-type product). Always
  written when `calculate_heliorv` is on, recording the RV *computed* for
  the exposure, regardless of whether it will actually be used downstream.
- **`HIERARCH WAVE RVCORR_APPLIED`** — written by `RSS.rectify_wave()`
  itself, in the final resampled product (e.g. the `hsci`-type product), and
  **only if** the correction it received was non-zero.

Because `resample_wavelength()` zeroes out `rv_corr` when
`apply_bary_corr=False` (`rssMethod.py`), running with the toggle off leaves
`BARYRV_APPLIED` present upstream (showing what *could* have been applied)
but `RVCORR_APPLIED` absent from the final file — so inspecting the final
product alone unambiguously answers "was this exposure's flux actually
corrected," without needing to know what CLI/config flags the run used.

### Deviation from the original plan

The plan called for exposing `apply_bary_corr` via both `etc/lvmdrp.yaml`
(generic `get_config_options` splat) and a CLI flag. In practice, adding it
to yaml as well would have caused `science_reduction(..., apply_bary_corr=X,
**kwargs)` to raise a duplicate-keyword error the moment someone also set it
in yaml (`run_drp` already threads `apply_bary_corr` as an explicit kwarg,
matching how `fluxcal_method` — the pattern being mirrored — actually works
today, which is *not* yaml-driven). Went with CLI + explicit kwarg only,
consistent with the existing `fluxcal_method` precedent; no yaml key was
added.

### Verification performed

- Sign convention verified numerically before writing any implementation
  code (see above), then locked in by a unit test.
- Full existing test suite: 65/65 passed (58 pre-existing + 7 new), run
  against the `lvmdrp26` conda environment (matching `master-perf`'s
  dependency versions) with `PYTHONPATH` pointed at this worktree.
- `ruff check` clean on all touched files.
- All touched modules import without error.

### Not yet done (needs real data / a running pipeline — out of reach from here)

- **Single real exposure smoke test**: run `create_pixel_table` →
  `resample_wavelength` on one real science exposure with
  `apply_bary_corr=True`, confirm `HIERARCH WAVE BARYRV_APPLIED` has a sane
  magnitude (well under ±35 km/s), and visually confirm sky-subtraction
  quality is unaffected relative to a run with the toggle off.
- **Cross-epoch combination check** (the actual motivating case): two
  exposures of the same field with substantially different barycentric
  corrections, confirm a spectral feature in the combined science target now
  lines up at the same output-grid wavelength in both.

## Open questions

Resolved during implementation:

- ~~Which coordinate defines the single per-exposure RV~~ — SCI field-center
  `SCIRA`/`SCIDEC` + `OBSTIME`, via `RSS.get_bary_rv()`.
- ~~Sign/direction convention~~ — verified numerically and locked in by a
  unit test (see "Implementation" above): `wave_native = rss._wave * (1 +
  rv_corr/c)`.
- ~~Toggle vs. always-on~~ — config/CLI toggle, default **ON**
  (`apply_bary_corr`, `--apply-bary-corr/--no-apply-bary-corr`).

Still open (not addressed by this change, tracked as follow-ups):

- Whether `rectify_wave`'s use of point interpolation instead of true
  flux-conserving rebinning is worth revisiting on its own — a related but
  separate, larger change from this feature.
- Whether `derive_vecshift(..., max_ampl=3)` in `model_selection()`
  (`fluxCalMethod.py:975`) needs to be widened to reliably capture standard
  star RV + injected correction combined (currently ≈±46 km/s total capture
  range, see above) — needs checking against the real RV spread of the
  standard-star catalog before deciding. A `# TODO` comment pointing at this
  was left next to the new correction code in `rss.py`, but the search
  window itself was not changed.
- The two "not yet done" real-data verification steps listed above
  (single-exposure smoke test, cross-epoch combination check).
