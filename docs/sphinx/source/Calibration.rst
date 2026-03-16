Calibration Processes
=====================

This document describes how master calibration products are created and applied
in the LVM DRP. Calibrations are produced separately from science reduction and
are referenced by the science pipeline via a calibration dictionary.

Overview
--------

The LVM DRP distinguishes between two kinds of calibration work:

1. **Master calibration creation** — processing dedicated calibration exposures
   (bias, dome flats, arc lamps, twilights) to produce reusable reference files.
   This is orchestrated by functions in ``lvmdrp/functions/run_calseq.py`` and
   invoked via the ``drp long-term`` or ``drp nightly`` CLI commands.

2. **Calibration application** — loading the pre-built master files and applying
   them to each science exposure during :func:`~lvmdrp.main.science_reduction`.
   No calibration is re-derived from the science frames themselves (except for a
   per-exposure thermal wavelength shift correction using sky lines; see
   :ref:`thermal-shift`).

Master calibrations are stored under::

    $SAS_BASE_DIR/sdsswork/lvm/calib/<mjd>/

and are selected at science-reduction time by :func:`~lvmdrp.utils.paths.get_calib_paths`,
which searches for the most appropriate epoch given the science MJD.

Calibration Types and Dependencies
-----------------------------------

The pipeline defines the following master calibration products, listed in
dependency order:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Product
     - File pattern
     - Description
   * - ``pixmask``
     - ``lvm-mpixmask-<camera>.fits``
     - Static bad-pixel mask for each detector
   * - ``pixflat``
     - ``lvm-mpixflat-<camera>.fits``
     - Pixel-level flat field (detector response)
   * - ``bias``
     - ``lvm-mbias-<camera>.fits``
     - Master bias frame (zero-level reference)
   * - ``centroids``
     - ``lvm-mcentroids-<camera>.fits``
     - Fiber centroid traces across the detector
   * - ``sigmas``
     - ``lvm-msigmas-<camera>.fits``
     - Fiber width (sigma) traces across the detector
   * - ``model``
     - ``lvm-mmodel-<camera>.fits``
     - Fiber amplitude traces (PSF model)
   * - ``wave``
     - ``lvm-mwave-<camera>.fits``
     - Wavelength solution trace (pixel → Angstrom)
   * - ``lsf``
     - ``lvm-mlsf-<camera>.fits``
     - Line Spread Function (FWHM in Å vs. wavelength)
   * - ``fiberflat_twilight``
     - ``lvm-mfiberflat_twilight-<channel>.fits``
     - Relative fiber-to-fiber throughput correction

Each calibration type depends on the products above it:

- ``bias`` requires ``pixmask``
- ``trace`` (centroids, sigmas, model) requires ``pixmask``, ``pixflat``, ``bias``
- ``wave`` / ``lsf`` require ``pixmask``, ``pixflat``, ``bias``, and all trace products
- ``fiberflat_twilight`` requires all of the above

Long-term vs. Nightly Calibrations
------------------------------------

Calibrations can be produced in two modes:

**Long-term calibrations** (default) are derived from a manually curated
reference epoch — typically one or two nights observed under photometric
conditions. They are identified by the ``lvm-m*`` filename prefix and are
reused across multiple science MJDs until conditions change significantly.

**Nightly calibrations** are derived fresh each night and use the ``lvm-n*``
prefix. They are selected when ``use_longterm_cals=False`` is passed to
:func:`~lvmdrp.main.science_reduction` or when the ``--nightly`` flag is used
with the CLI.

Both modes produce identical calibration products; the distinction is in their
cadence and reuse policy.

Master Bias
-----------

**Function**: :func:`~lvmdrp.functions.run_calseq.create_bias`

**Inputs**: A set of bias exposures (typically 7–9 frames) taken at the
calibration epoch.

**Process**:

1. Each bias frame is preprocessed (overscan subtraction, pixel masking) and
   detrended.
2. The individual frames are combined with sigma-clipping into a single
   ``lvm-mbias-<camera>.fits`` file per detector (9 cameras total: b1–b3,
   r1–r3, z1–z3).

**Output**: Master bias frames used in subsequent detrending steps.

Fiber Traces
------------

**Functions**: :func:`~lvmdrp.functions.run_calseq.create_traces` /
:func:`~lvmdrp.functions.run_calseq.create_nightly_traces`

**Inputs**: Dome flat exposures illuminated by two different continuum lamps —
LDLS (broad-spectrum, used for b/r channels) and Quartz (used for z channel).
Typically 24 frames total (12 per lamp type) are combined.

**Process**:

1. Dome flats are 2D-reduced (preprocessed and detrended using the master bias
   and pixel flat).
2. Fiber centroids are located along the cross-dispersion axis and traced across
   the full CCD using polynomial fits (degree 4 by default).
3. Fiber widths (sigma of the cross-dispersion profile) are traced similarly
   (degree 8 polynomial).
4. A PSF amplitude model is also traced (degree 5 polynomial).

**Outputs**:

- ``lvm-mcentroids-<camera>.fits`` — fiber centroid positions vs. column
- ``lvm-msigmas-<camera>.fits`` — fiber width vs. column
- ``lvm-mmodel-<camera>.fits`` — fiber amplitude model vs. column

These traces are used during science reduction for stray light subtraction
(centroids only) and 1D spectral extraction (all three).

Wavelength Solution and LSF
----------------------------

**Functions**: :func:`~lvmdrp.functions.run_calseq.create_wavelengths`, which calls
:func:`~lvmdrp.functions.rssMethod.determine_wavelength_solution`

**Inputs**: Arc lamp exposures (typically 24 frames: 12 at 10 s exposure and
12 at 50 s) covering all three spectral channels. Multiple lamp types are used
to ensure adequate line coverage across the full wavelength range.

**Process**:

1. Arc frames are 2D-reduced and 1D-extracted using the master fiber traces.
2. A reference fiber spectrum is used to identify known arc lines from a
   pre-defined line list (pixel–wavelength map stored in ``lvmcore``).
3. For each fiber, Gaussian profiles are fitted to the arc lines. This yields
   the line centroid (in wavelength, for the wavelength solution) and the line
   FWHM (in Angstroms, for the LSF).
4. A polynomial (default degree 5 in dispersion) is fitted per fiber to
   map CCD pixel position to wavelength. Polynomial coefficients are stored in
   a :class:`~lvmdrp.core.tracemask.TraceMask` object.
5. A separate polynomial (default degree 2) is fitted per fiber to the
   measured FWHM as a function of wavelength. This is the Line Spread Function
   (LSF) — the Gaussian width of the instrumental profile at each wavelength.
6. Coefficients for both solutions are interpolated across fibers where fitting
   failed (e.g., dead or low-S/N fibers).

**Outputs**:

- ``lvm-mwave-<camera>.fits`` — per-fiber wavelength solution as polynomial
  coefficients and evaluated pixel table
- ``lvm-mlsf-<camera>.fits`` — per-fiber LSF (FWHM in Å) as polynomial
  coefficients and evaluated pixel table

**Application to science**: During science reduction,
:func:`~lvmdrp.functions.rssMethod.create_pixel_table` loads these master files
and evaluates the polynomial coefficients at each fiber's native pixel positions
to produce a 2D array (nfibers × npixels) of wavelength and FWHM values. During
wavelength resampling, the LSF array is linearly interpolated onto the new common
wavelength grid and propagated through to the final ``lvmCFrame`` product as the
``LSF`` FITS extension.

.. _thermal-shift:

Thermal Wavelength Shift Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The arc-derived wavelength solution is a static calibration and does not
account for thermal drifts or mechanical flexure between the calibration epoch
and a given science exposure. A per-exposure zero-point correction is applied
during science reduction by
:func:`~lvmdrp.functions.rssMethod.shift_wave_skylines`.

The correction procedure is as follows:

1. For each fiber individually, Gaussian profiles are fitted to one or more
   known sky emission lines (e.g., [O I] 5577 Å). The difference between each
   fitted line centroid and its known vacuum wavelength gives a per-fiber,
   per-line offset in Angstroms. Fibers that are dead, belong to the Spec
   telescope, or have S/N < 10 around the sky lines are skipped.

2. The per-fiber offsets are smoothed along the slit within each spectrograph
   separately: a median filter is applied first, then a spline is fitted to the
   surviving fibers. This produces a smooth ``fiber_offset_mod`` array that
   varies continuously across the 648 fibers of each spectrograph.

3. Only the **zeroth polynomial coefficient** of the stored wavelength trace is
   adjusted (``wave_trace['COEFF'][:,0] -= fiber_offset_mod``). The higher-order
   coefficients — which encode how wavelength varies with pixel position along
   the dispersion axis — are left unchanged from the arc calibration.

The consequence is that the *shape* of the wavelength solution (spectral
sampling, relative pixel spacing) is entirely determined by the arc lamps.
The sky-line step corrects only the zero-point of each fiber's wavelength
array, accounting for any rigid shift since the arc was taken. It does not
re-derive or re-fit the full wavelength polynomial.

**Reference sky lines** (``REF_SKYLINES`` in ``core/constants.py``, current as
of this documentation):

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Channel
     - Wavelengths (Å, vacuum)
     - Notes
   * - b
     - 5577.35
     - [O I]; sole bright isolated line in the blue channel
   * - r
     - 6363.78, 7358.68, 7392.21
     - [O I]; OH, OH
   * - z
     - 8399.18, 8988.38, 9552.55, 9719.84
     - OH, OH, OH, OH

Lines were hand-picked as bright, isolated (non-doublet) features in the UVES
sky atlas. The per-fiber offset is the median over all lines in the channel.

Twilight Fiberflat
------------------

**Function**: :func:`~lvmdrp.functions.run_calseq.create_twilight_fiberflats`

**Inputs**: Twilight sky exposures taken near dawn or dusk (typically 12–24
frames per epoch). Because the twilight sky is a smooth, near-featureless
continuum, it provides a clean measure of the relative throughput of each
fiber at the time of the calibration sequence.

**Process**:

1. Twilight frames are 2D-reduced, 1D-extracted, and wavelength-calibrated
   using the master traces and wave/LSF solutions described above.
2. Spectra are resampled onto the common 0.5 Å wavelength grid for each
   channel (b, r, z).
3. A reference spectrum is formed (default: the ``nanmedian`` across all
   fibers). Each fiber is divided by the reference to measure its relative
   throughput as a function of wavelength.
4. The per-fiber throughput curves are smoothed spatially (across the slit)
   using a polynomial fit and a spline or polynomial smoothing kernel to
   suppress noise while preserving real large-scale throughput gradients.
5. Dead or low-S/N fibers are interpolated from their neighbours.
6. Normalisation is anchored to a reference wavelength (e.g., the [O I]
   5577 Å sky line for the blue channel) to set the absolute scale.

**Outputs**:

- ``lvm-mfiberflat_twilight-<channel>.fits`` (one per channel: b, r, z)

**Application to science**: Applied by
:func:`~lvmdrp.functions.rssMethod.apply_fiberflat` to the
wavelength-calibrated science RSS to remove fiber-to-fiber throughput
variations before sky subtraction.

Science Fiberflat Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An optional refinement step,
:func:`~lvmdrp.functions.run_calseq.create_fiberflats_corrections`, uses sky
emission lines detected in a set of science frames to correct residual
large-scale throughput errors in the master twilight fiberflat. The correction
is derived by fitting the ratio of the per-fiber sky line flux to the median,
sigma-clipping outliers, and multiplying the master fiberflat by the resulting
smooth correction map. The master fiberflat is updated in place.

Running the Calibration Pipeline
---------------------------------

Master calibrations can be created with the ``drp`` CLI::

    # Run the long-term calibration sequence for a given MJD
    drp long-term -m <mjd>

    # Run a nightly calibration sequence
    drp nightly -m <mjd>

    # Run individual calibration steps
    drp create-bias -m <mjd>
    drp create-traces -m <mjd>
    drp create-wavelengths -m <mjd>
    drp create-fiberflats -m <mjd>

Pre-built master calibrations can be downloaded from the SAS::

    drp get-calibs -m <mjd>

Calibration Selection During Science Reduction
------------------------------------------------

When :func:`~lvmdrp.main.science_reduction` is invoked, it calls
:func:`~lvmdrp.utils.paths.get_calib_paths` to locate the appropriate master
calibration files for the science MJD. The function searches backward in time
from the science MJD to find the most recent available calibration epoch.
Long-term calibrations are preferred by default; passing
``use_longterm_cals=False`` selects nightly calibrations instead.

The resolved calibration paths are assembled into a dictionary with the
following structure::

    calibs = {
        "pixmask":            {"b1": path, "b2": path, ..., "z3": path},
        "pixflat":            {"b1": path, ..., "z3": path},
        "bias":               {"b1": path, ..., "z3": path},
        "centroids":          {"b1": path, ..., "z3": path},
        "sigmas":             {"b1": path, ..., "z3": path},
        "model":              {"b1": path, ..., "z3": path},
        "wave":               {"b": [path_b1, path_b2, path_b3], "r": [...], "z": [...]},
        "lsf":                {"b": [path_b1, path_b2, path_b3], "r": [...], "z": [...]},
        "fiberflat_twilight": {"b": path, "r": path, "z": path},
    }

Each science exposure uses the same ``calibs`` dictionary; no per-exposure
calibrations are derived from the science frames themselves.
