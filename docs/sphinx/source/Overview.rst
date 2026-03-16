Pipeline Overview
=================

This document describes the main steps of the LVM Data Reduction Pipeline (DRP),
as implemented in ``lvmdrp/main.py``.

The pipeline processes raw spectroscopic data from the Local Volume Mapper (LVM)
instrument and produces fully calibrated, sky-subtracted spectra. The main entry
point is :func:`~lvmdrp.main.run_drp`, which orchestrates the full reduction through
:func:`~lvmdrp.main.science_reduction`.

Pipeline Stages
---------------

The reduction is organized into three main stages that can be controlled via
command-line flags (``skip_2d``, ``skip_1d``, ``skip_post_1d``):

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

**Stray Light Subtraction** (:func:`~lvmdrp.functions.imageMethod.subtract_straylight`):

- Models and removes scattered light contamination
- Uses fiber centroid traces to mask fibers before fitting the stray light field

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

**Thermal Shift Correction** (:func:`~lvmdrp.functions.rssMethod.shift_wave_skylines`):

- Refines wavelength calibration using sky emission lines
- Corrects for thermal drifts since the arc lamp calibration

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

Flux calibration converts instrumental counts to physical flux units:

**Model Selection** (:func:`~lvmdrp.functions.fluxCalMethod.model_selection`):

- Selects appropriate stellar atmosphere models for calibration stars
- Uses cached Gaia XP spectra and Pollux library models

**Standard Star Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.fluxcal_standard_stars`):

- Derives sensitivity curves from standard stars observed in the spec telescope
- Uses Gaia XP spectra as reference

**Science IFU Star Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.fluxcal_sci_ifu_stars`):

- Additional flux calibration using field stars in the science IFU
- Provides spatial calibration across the field

**Apply Flux Calibration** (:func:`~lvmdrp.functions.fluxCalMethod.apply_fluxcal`):

- Applies the derived sensitivity curve to all fibers
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

7. Summary Generation
^^^^^^^^^^^^^^^^^^^^^

**Update Summary File** (:func:`~lvmdrp.utils.metadata.update_summary_file`):

- Updates the drpall summary file with metadata from the reduction
- Includes quality metrics and observation parameters

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

Running the Pipeline
--------------------

The pipeline can be run via the command line::

    # Reduce all exposures for an MJD
    drp run -m <mjd>

    # Reduce a single exposure
    drp run -e <expnum>

Optional flags control which stages are executed::

    --skip-2d        Skip preprocessing and detrending
    --skip-1d        Skip fiber extraction
    --skip-post-1d   Skip wavelength calibration through sky subtraction

See the CLI documentation for additional options including flux calibration
methods and calibration file selection.
