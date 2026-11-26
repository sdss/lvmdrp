Pipeline Overview
=================

This document describes the main steps of the LVM Data Reduction Pipeline (DRP),
as implemented in ``lvmdrp/main.py``.

The pipeline processes raw spectroscopic data from the Local Volume Mapper (LVM)
instrument and produces fully calibrated, sky-subtracted spectra. The main entry
point is :func:`~lvmdrp.main.run_drp`, which orchestrates the full reduction through
:func:`~lvmdrp.main.science_reduction`.

Before the Reduction
--------------------

For each exposure, :func:`~lvmdrp.main.science_reduction` first:

- updates the frames metadata of the night (``update_metadata``) and reads the exposure's
  record from it
- skips the exposure if the raw-data quality pipeline flagged it BAD (``qaqual``), unless
  ``--force-run`` is given; ``run_drp()`` also skips exposures listed in
  ``$LVMCORE_DIR/etc/exclude-exposures.yaml``, unless ``--force-excluded`` is given
- selects the master calibrations of its calibration epoch, the closest epoch at or before
  the exposure's MJD (``get_master_mjd``; the epochs are defined in
  ``$LVMCORE_DIR/calibrations/calibration-epochs.yaml``): pixel mask, pixel flat, bias,
  fiber traces (centroids, sigmas, model), wavelength and LSF, and the twilight fiberflat,
  read from ``$LVM_MASTER_DIR``

How the master calibrations are made, selected and applied is described in
:doc:`Calibration`.

Pipeline Stages
---------------

The reduction runs in the stages below. Each group of steps can be skipped with a
``drp run`` flag, e.g. to redo only the later steps of an exposure already reduced:

======================  ==========================================================
Flag                    Steps skipped
======================  ==========================================================
``-2d``                 preprocessing, detrending, astrometry, stray light
``-1d``                 fiber extraction
``-wc``                 spectrograph stacking, wavelength calibration, fiberflat,
                        thermal shift correction
``-wr``                 sky fiber interpolation, sky combination, wavelength
                        resampling
``-fl``                 flux calibration and channel combination
``-sk``                 sky subtraction
``-da``                 drpall summary update
======================  ==========================================================

1. 2D Reduction
^^^^^^^^^^^^^^^

The 2D reduction stage (:func:`~lvmdrp.main.reduce_2d`) processes the raw CCD images:

**Preprocessing** (:func:`~lvmdrp.functions.imageMethod.preproc_raw_frame`):

- Overscan subtraction
- Application of pixel masks to flag bad pixels
- Replacement of flagged pixels with NaN values

**Detrending** (:func:`~lvmdrp.functions.imageMethod.detrend_frame`):

- Bias subtraction using master bias frames
- Pixel flat correction
- Cosmic ray rejection and flagging

**Astrometry** (:func:`~lvmdrp.functions.astrometryMethod.add_astrometry`):

- Adds astrometric solution to detrended frames using guide camera coadds
- Processes science, SkyE, and SkyW telescope pointings
- For each telescope, :func:`~lvmdrp.core.astrometry.set_telescope_astrometry`
  writes the ``SCIRA``/``SCIDEC``, ``SKYERA``/``SKYEDEC``, and
  ``SKYWRA``/``SKYWDEC`` keywords (plus ``ALT``/airmass via
  :func:`~lvmdrp.core.astrometry.set_altaz_params`) that the rest of the DRP
  and the drpall summary file rely on (see :doc:`SummaryFile`). These are
  derived one of two ways, recorded in the ``<TEL>ASRC`` keyword
  (``SCIASRC``/``SKYEASRC``/``SKYWASRC``):

  - ``'GDR coadd'`` — fit independently from that telescope's own guide
    camera coadd WCS solution (used whenever the coadd has ``SOLVED=True``).
  - ``'CMD position'`` — a fallback that copies the *commanded* pointing
    straight from the raw header (``POSCIRA``/``POSKYERA``/``POSKYWRA``,
    etc.) when no guider astrometric solution is available. This path does
    not independently verify telescope identity — if the commanded
    coordinates for SkyE/SkyW were already swapped upstream (e.g. by the
    telescope-control/scheduling system, before the file reaches lvmdrp),
    this fallback will silently propagate that swap into ``SKYERA``/``SKYWRA``
    and everything derived from them. Checking ``<TEL>ASRC`` for an affected
    exposure shows which path was used.

**Stray Light Subtraction** (:func:`~lvmdrp.functions.imageMethod.subtract_straylight`):

- Models and removes scattered light contamination
- Uses fiber centroid traces to mask fibers, then samples the remaining
  inter-fiber background in strips across the CCD
- Fits a 2D tensor-product P-spline surface to the background samples, posed
  and solved as a quadratic program (via ``osqp``) with independent
  smoothness penalties along the dispersion and cross-dispersion axes
  (:meth:`~lvmdrp.core.image.Image.fit_pspline2d`)
- Iterates with sigma-clipping to reject outlying pixels before the final fit
  (:meth:`~lvmdrp.core.image.Image.fit_pspline2d_iterative`)

2. 1D Extraction
^^^^^^^^^^^^^^^^

The 1D extraction stage converts the 2D CCD images to row-stacked spectra (RSS):

**Fiber Extraction** (:func:`~lvmdrp.functions.imageMethod.extract_spectra`):

- Extracts 1D spectra from each fiber using the fiber trace centroids
- Supports optimal extraction (default) or aperture extraction methods
- Uses pre-computed trace centroids, widths, and PSF models from master calibrations

3. Post-1D Processing
^^^^^^^^^^^^^^^^^^^^^

The post-1D stage performs spectral calibration and combination. Operations are
performed per spectral channel (b, r, z) before final combination:

**Spectrograph Stacking** (:func:`~lvmdrp.functions.rssMethod.stack_spectrographs`):

- Combines the three spectrographs (sp1, sp2, sp3) for each channel
- Produces a single RSS file per channel with all 1944 fibers (648 per spectrograph)

**Wavelength Calibration** (:func:`~lvmdrp.functions.rssMethod.create_pixel_table`):

- Applies the wavelength solution derived from arc lamp exposures
- Assigns wavelength values to each pixel in each fiber

**Fiberflat Correction** (:func:`~lvmdrp.functions.rssMethod.apply_fiberflat`):

- Corrects for fiber-to-fiber throughput variations
- Uses master fiberflat derived from twilight observations
- Spectrograph-to-spectrograph residual factors are no longer fit on the fly
  here; they are pre-measured from science exposures and baked into the
  master fiberflat ahead of time (see :ref:`fiberflat-refinement`). The IFU
  gradient fitted together with the factors is also applied, to the science
  fibers only (header keyword ``<ARM> FIBERFLAT GRADAPP``). This step warns if
  the master fiberflat has not received that correction

**Thermal Shift Correction** (:func:`~lvmdrp.functions.rssMethod.shift_wave_skylines`):

- Refines wavelength calibration using sky emission lines
- Corrects for thermal drifts since the arc lamp calibration (see :ref:`thermal-shift`)

**Sky Fiber Interpolation** (:func:`~lvmdrp.functions.skyMethod.interpolate_sky`):

- Interpolates sky spectra from dedicated sky fibers
- Prepares sky model for each science fiber

**Sky Combination** (:func:`~lvmdrp.functions.skyMethod.combine_skies`):

- Combines sky measurements from east and west sky telescopes
- Applies configurable weights to each telescope

**Wavelength Resampling** (:func:`~lvmdrp.functions.rssMethod.resample_wavelength`):

- Resamples all spectra onto a uniform wavelength grid
- Uses 0.5 Angstrom dispersion
- Converts to spectral density units (per Angstrom)

4. Flux Calibration
^^^^^^^^^^^^^^^^^^^

Flux calibration converts instrumental counts to physical flux units.

The method is chosen with ``drp run -f``: ``MOD`` (default) uses the standard stars
matched to model spectra, ``STD`` the standard stars, ``SCI`` the field stars in the
science IFU, and ``NONE`` doesn't apply any. Sensitivities are measured for all of
them, so the choice only changes which one is applied.

**Model Selection** (:func:`~lvmdrp.functions.fluxCalMethod.model_selection`):

- Selects appropriate stellar atmosphere models for calibration stars
- Uses cached Gaia XP spectra (``$LVM_MASTER_DIR/gaia_cache``) and Pollux library models
- Retrieves Gaia DR3 stellar physical parameters (:math:`T_\mathrm{eff}`,
  :math:`\log g`, [M/H]) for each standard star via
  :func:`~lvmdrp.core.fluxcal.get_stellar_params`, preferring a local
  calibration table and falling back to a live Gaia query

**Standard Star Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.fluxcal_standard_stars`):

- Derives sensitivity curves from standard stars observed in the spec telescope
- Uses Gaia XP spectra as reference, retrieved and cached through the
  :class:`~lvmdrp.core.fluxcal.GaiaXPSpectra` class

**Science IFU Star Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.fluxcal_sci_ifu_stars`):

- Additional flux calibration using field stars in the science IFU
- Provides spatial calibration across the field

**Apply Flux Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.apply_fluxcal`):

- Applies the sensitivity of the chosen method to all fibers, with a telluric correction
- Produces flux-calibrated frames (FFrame) for each channel

5. Channel Combination
^^^^^^^^^^^^^^^^^^^^^^

**Join Spectral Channels** (:func:`~lvmdrp.functions.rssMethod.join_spec_channels`):

- Stitches the b, r, z channels into a single continuous spectrum
- Handles overlap regions with weighted combination
- Produces the combined frame (CFrame)

6. Sky Subtraction
^^^^^^^^^^^^^^^^^^

**Quick Sky Subtraction** (:func:`~lvmdrp.functions.skyMethod.quick_sky_subtraction`):

- Subtracts the interpolated sky model from science fibers
- Produces the final sky-subtracted frame (SFrame)

.. _summary-generation:

7. Summary Generation
^^^^^^^^^^^^^^^^^^^^^

**Update Summary File** (:func:`~lvmdrp.utils.metadata.update_summary_file`):

- Updates the drpall summary file with metadata from the reduction
- Includes quality metrics and observation parameters
- See :doc:`SummaryFile` for a full description of the drpall file structure
  and a table mapping each column to its source ``lvmSFrame`` header keyword
- Batch runs (``drp cluster``) skip this step in every job; the summary is then made
  once with ``drp summary``

Output Data Products
--------------------

The pipeline produces several intermediate and final data products:

**Intermediate Products** (in ``ancillary/`` directory):

- ``lvm-pobject-*.fits`` - Preprocessed frames
- ``lvm-dobject-*.fits`` - Detrended frames
- ``lvm-lobject-*.fits`` - Stray light subtracted frames
- ``lvm-xobject-*.fits`` - Extracted spectra
- ``lvm-wobject-*.fits`` - Wavelength calibrated spectra
- ``lvm-sobject-*.fits`` - Sky-interpolated spectra
- ``lvm-hobject-*.fits`` - Resampled spectra

**Final Products**:

- ``lvmFrame-{channel}-{expnum}.fits`` - Extracted, wavelength calibrated per channel
- ``lvmFFrame-{channel}-{expnum}.fits`` - Flux calibrated per channel
- ``lvmCFrame-{expnum}.fits`` - Channel combined (full wavelength coverage)
- ``lvmSFrame-{expnum}.fits`` - Sky subtracted (final science product)

.. _running-the-pipeline:

Running the Pipeline
--------------------

Running the DRP needs the environment set up as described in :doc:`Installation`, and
the master calibrations of the epochs to be reduced (``drp get-calibs``; see
:doc:`Calibration`).

To reduce all the science exposures of a night, one exposure, or several::

    drp run -m <mjd>
    drp run -e <expnum>
    drp run -el <expnum> -el <expnum>          # a list
    drp run -er <first>-<last>                 # a range
    drp run -F <expnum_file>                   # a file with one exposure number per line

Steps already done can be skipped with the flags in the table under `Pipeline Stages`_.
For example, to redo only the flux calibration, sky subtraction and summary of an
exposure already reduced::

    drp run -e <expnum> -2d -1d -wc -wr

Other useful options:

- ``-f <method>``: the flux calibration applied to the science fibers: ``MOD`` (default;
  standard stars matched to model spectra), ``STD`` (standard stars), ``SCI`` (field stars
  in the science IFU) or ``NONE``
- ``-c``, ``--clean-ancillary``: remove the intermediate products (the ``ancillary``
  folder) once the exposure is done
- ``--force-run``: also reduce exposures flagged BAD by the raw-data QA
- ``--force-excluded``: also reduce exposures listed in
  ``$LVMCORE_DIR/etc/exclude-exposures.yaml``

``drp run --help`` lists all the options.

The products are written to
``$SAS_BASE_DIR/sdsswork/lvm/spectro/redux/<drpver>/<tilegrp>/<tileid>/<mjd>/`` (see
`Output Data Products`_ and :doc:`DirectoryStructure`). ``<drpver>`` is
``$LVMDRP_VERSION`` if it is set, otherwise the version of the installed package, so
setting it keeps test reductions apart::

    export LVMDRP_VERSION="my_tests"

Batch Runs at Utah
^^^^^^^^^^^^^^^^^^

At Utah, ``drp cluster`` writes and submits a Slurm job that runs ``drp run`` for each
night or exposure given; ``--drp-options`` passes options on to ``drp run``::

    drp cluster -F <expnum_file> --drp-options "<options for drp run>"
    drp cluster -r <first_mjd>-<last_mjd> -n <nodes> -p <cores_per_node> -w <walltime>

``--dry-run`` shows what would be run. Each job runs with ``--skip-drpall``, so that the
jobs don't write the summary file at the same time. Once the jobs are done, make the
summary file in one go::

    drp summary --drp-version <drpver>

``--output-dir`` writes it elsewhere, e.g. without write access to the redux directory.
The master calibrations of several epochs can be made as one Slurm job in the same way
(``drp calibrations cluster``; see :doc:`Calibration`).

Checking a Reduction
--------------------

``drp qa-science -e <expnum>`` checks the reduction of an exposure step by step and writes
an HTML dashboard: detrending, stray light, fiber and sky-line shifts, the IFU, flux
calibration and sky subtraction. It reuses the measurements stored in the products.
