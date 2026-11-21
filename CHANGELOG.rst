.. _drp-changelog:

==========
Change Log
==========

This document records the main changes to the drp code.

1.3.3dev (unreleased)
---------------------

- Bumped ``skycalc_cli`` to 1.5, which no longer imports the deprecated ``pkg_resources``; removes the setuptools warning on every ``drp`` call and the ``ModuleNotFoundError`` with setuptools>=81 (#257)
- Replaced lvmdrp's own ``pkg_resources.parse_version`` with ``packaging.version.parse``
- Optimal extraction is about 5x faster with identical results: ``FiberProfileCache`` now reads the two cumulative sums each pixel needs directly from the profile cache, instead of copying every oversampled row of the selected profiles for each column, which took about 80% of the extraction time (about 40% of a science reduction, and most of the arc and twilight extraction in the calibration sequence)
- Cosmic-ray rejection (``Image.reject_cosmics``) is 10-20% faster with identical results: the Laplacian of the input image, which does not change between iterations, is computed once instead of in each of the 5 iterations
- Stray-light subtraction is about 2.5x faster with identical results: its QA figure (``fit_straylight``), which took about 80% of the time, now draws about 400 rows and columns of the smooth model instead of all of them, and the per-pixel error bars of the fitted strips as one band per column
- LSF matching of the twilight and dome flats (``match_resolution``) is about 20x faster with identical results: ``convolution_matrix`` builds the sparse convolution matrix with array operations instead of a Python loop over every matrix entry, and ``flatten_lsf`` builds its Gaussian kernels in one array operation; this step took about 6 minutes of each twilight exposure in the calibration sequence
- ``drp``, ``pix2wave`` and ``envcheck`` are now ``console_scripts`` entry points (code moved from ``bin/`` to ``lvmdrp.cli``) instead of scripts: pip copied scripts into the environment at install time, so an editable install kept running an old ``drp`` against newer library code after a ``git pull`` or ``git checkout`` (e.g. ``ImportError: cannot import name ...``) until it was reinstalled
- Added ``functions.flatgradient`` to measure the large-scale flat-field error of the science IFU from sky lines in lvmCFrames: spectrograph factors and a plane per line (5577, 6364, 8399 and held-out OH lines), and the spectrograph factors in the sky IFUs; the results are written as ``SKYPLANE`` header keywords of every lvmCFrame made by ``science_reduction`` (a failed measurement is logged and does not stop the reduction), and copied to the drpall as ``skyplane_*`` columns (-999 for frames without them) (#268)
- Added ``drp calibrations fit-fiberflat-gradient``, which fits the model of the large-scale flat-field error (spectrograph factors and a plane across the science IFU, with a constant and drift per ffactor epoch and shared focus and K-mirror terms) to the ``skyplane_*`` columns of a drpall file and writes one ``lvm-mfiberflat_gradient-[CHANNEL].fits`` calibration file per ffactor epoch (#268)
- ``science_reduction`` now corrects the science fibers of each lvmCFrame for the large-scale flat-field error, when the ``lvm-mfiberflat_gradient`` files of its ffactor epoch exist: the model's spectrograph factors and plane, evaluated from MJD, TESCIFO and TESCIKM, optionally followed by a direct fit of what is left (``--flatgrad-direct``); recorded as ``FLATGRAD`` header keywords, never applied twice, and skipped with ``--skip-flatgrad``. The ``SKYPLANE`` measurement is made before the correction (#268)
- Added ``drp run --from-cframe``, which starts from the existing lvmCFrames (e.g. of a released reduction, with ``LVMDRP_VERSION``): it measures the sky-line planes, applies the large-scale flat-field correction, replaces the lvmCFrame and redoes the sky subtraction and the drpall update; with ``--new-cframe`` the corrected frames are written as new ``.flatgrad.fits`` files and the originals and the drpall are left as they are, for testing (#268)
- ``quick_sky_subtraction`` creates the ``ancillary`` and ``qa`` directories when they are missing
- Added ``drp calibrations twilight-gradient``, which fits the gradient removed from each twilight exposure (from the ``qa-twilights --trends`` table) as a constant plane per ffactor epoch plus a twilight sky gradient towards the Sun, and puts each epoch's plane into the science fibers of its master twilight flats; by default it writes them into a new calibration directory of symbolic links (``--out-calib-dir``), leaving the original flats as they are, and changes them only with ``--in-place``; ``--undo`` takes the gradient out again (#268)
- Added ``drp run --calib-dir`` to reduce with a calibration directory other than ``$LVM_MASTER_DIR``, taking precedence over the environment (which a calling tool's sdss tree setup can reset); the calibration directory is now read when needed (``get_masters_dir()``) rather than at import, and recorded in the ``CALIBDIR`` header keyword and the drpall ``calibdir`` column (#272)
- ``lvmdrp.yaml`` can name a calibration directory (``calib_dir``, relative to ``$SAS_BASE_DIR/sdsswork/lvm/`` or absolute) and a version label (``drpver``), so that a branch carries its own calibration set and label and checking it out selects them. Every ``drp`` command uses ``calib_dir`` instead of ``$LVM_MASTER_DIR`` unless ``--calib-dir`` is given, stops if the directory doesn't exist rather than falling back to ``$LVM_MASTER_DIR``, and logs the label and calibration directory when it starts; ``$LVMDRP_VERSION`` still takes precedence over ``drpver``. Both are empty by default, which changes nothing (#272)
- Record the lvmcore commit used (``-dirty`` if its tracked files were changed) in the log, the ``LVMCORE`` header keyword and the drpall ``lvmcore`` column, since the calibration epochs, excludes and header fixes in lvmcore change independently of lvmdrp (#272)
- Record the lvmdrp commit used (``-dirty`` if its tracked files were changed, ``unknown`` outside a git checkout) in the log, the ``LVMDRP`` header keyword and the drpall ``lvmdrp`` column, since ``DRPVER`` is only a version label; ``CALIBDIR``, ``LVMCORE`` and ``LVMDRP`` are set by one function, ``constants.add_provenance`` (#268, #272)
- The master twilight and dome fiber flats now list the camera frames they were combined from in a ``FRAMES`` extension (the same columns as the other masters: TILEID, MJD (the SJD), EXPNUM, SPEC, CAMERA, EXPTIME) and the number of exposures in ``NFRAMES``; stage 2 (``do_ffactor_correction``, ``undo_ffactor_correction``) and ``twilight-gradient`` keep it when they rewrite a flat. The master centroids and sigmas now also list the traced exposures in ``FRAMES``, so that every master records its exposures (#268)
- Added ``drp run --from-frame``, which starts from the existing lvmFrames and corrects every fiber of each channel for the large-scale flat-field error (the spectrograph factors on all fibers, the plane on the science fibers only), then redoes the sky interpolation, resampling, flux calibration, joining, sky subtraction and the drpall update. The correction comes from the sky-line planes in the exposure's lvmCFrame and the ``lvm-mfiberflat_gradient`` model of its epoch where one exists (``--flatgrad-direct`` uses the exposure's own measurements, with the b/z guard when there is a model); it is recorded as ``FFCORR`` keywords in the lvmFrame headers, the SUPERFLAT extension stays the flat applied, and a rerun replaces the earlier correction rather than adding to it (#268)
- ``drp calibrations fit-fiberflat-gradient`` can fit per calibration epoch (``--epochs calibration``), writing each model next to that epoch's twilight flats, where ``drp run --from-frame`` looks for it; can hold the shared focus and K-mirror terms fixed at those of existing model files (``--shared-from DIR``) and fit only the epoch terms; and can fit fewer epoch terms (``--epoch-terms c0,t1``) when an epoch has only a few nights (#268)
- The ``FFCORR`` and ``FLATGRAD`` corrections are copied to the drpall as ``ffcorr_*`` and ``flatgrad_*`` columns (-999 or ``None`` for frames not corrected) (#268)
- The ``lvm-mfiberflat_gradient`` model files record how they were made (``DRPALL``, the drpall file fitted; ``EPTERMS``, ``MINEXP``, ``FITDATE``, and the ``LVMDRP`` and ``LVMCORE`` commits) and carry an identifier, ``MODELID``, a hash of the coefficients. Since a refit keeps the file name, every corrected frame records the identifier of the model used next to its file name (``FFCORR <CH> MODELID``, ``FLATGRAD <CH> MODELID``), and both go to the drpall (``ffcorr_<ch>_file``, ``ffcorr_<ch>_modelid`` and the ``flatgrad_*`` equivalents) (#268)
- ``drp calibrations fiberflats -e EPOCH`` makes the master twilight flats of whole calibration epochs from their twilight nights in the epochs file, after linking the masters stage 1 needs from the calibration directory in use into the reduction tree; ``-m`` without an epochs table took the twilight nights from the ``-m`` date (#268)
- Added ``drp calibrations build-calib-set``, which builds or updates a calibration set: symbolic links to every file of an existing set (by default ``sandbox/calib``), the Gaia cache shared, and with ``--flats-from`` the stage-1 twilight flats of a reduction copied in place of their links; the source set is never changed and each run is recorded in ``README.txt`` (``lvmdrp.utils.calibset``). ``link_calibration_tree`` now also links directories that are themselves links (#268, #272)
- ``drp calibrations fit-fiberflat-gradient`` needs to be told where the models go (``--out-dir``, ``--to-redux`` or ``--to-calib-dir``) instead of writing into ``$LVM_MASTER_DIR`` by default, where every reduction with that calibration set would apply them (#268)
- The sky-line measurement rejects stars again after flattening the continuum by the first fit, since in a frame with a large gradient the continuum percentiles otherwise reject ordinary fibers on one side of the IFU (a bias of 0.1-0.3 %/10 mm); ``measure_cframe`` leaves out the masked pixels, which the written lvmCFrame has as NaN, so that the measurement is that of the product (#268)


1.3.2 (02-09-2026)
------------------

- Fixed ancillary cleanup so QA output paths are skipped during reduction housekeeping
- Corrected Ruff/CI configuration and lint-related defaults to match project expectations

1.3.1 (25-08-2026) [YANKED]
---------------------------

Superseded by 1.3.2. Ancillary products would leave a large footprint on disk per catchup run.
Do not use this version.

- Fixed Gaia XP cache directory during flux calibration

1.3.0 (24-08-2026) [YANKED]
---------------------------

Superseded by 1.3.1. Broken Gaia caching prevented any reduction from running.
Do not use this version.

- Improved QA plots including fiber shift, sky parameters, and layout handling
- Implemented fiber flat factor epochs and improved factor calibration routines
- Enhanced straylight subtraction with robust 2D P-Spline fitting
- Improved Gaia XP integration with better caching and stellar parameter extraction
- Implemented Voigt profile for fiber tracing with better convergence
- Performance optimizations including memory usage cleanup
- Added RMS sensitivities to summary file
- Improved outlier rejection in stray light binning and saturated strip handling
- Enhanced flux calibration with better handling of standard star selection
- Various bug fixes and stability improvements

1.2.1 (16-02-2026)
------------------

- Implemented tellurics correction
- Fixed issues with wavelength solutions
- Fixed issues with straylight subtraction
- Fixed bugs in astrometric solutions
- Added flux calibration summary to drpall
- First stab at documentation
- A few more bug fixes and improvements

1.2.0 (15-11-2025)
------------------

- Fitting routines now perform pixel integration
- Physically motivated fiber profile for fiber modeling
- Extraction using new fiber profile and speed up
- New fiber flatfielding algorithm to account for shutter timing effects
- Fixes resampling bug to get back 2% in flatfielding
- New flux calibration using template matching
- Fixes multiple stars issue in science field flux calibration
- Much improved calibrations pipeline
- Improved CR masking
- Many more bug fixes and improvements

1.1.1 (20-12-2024)
------------------

- Honor MJD exclusion list stored in LVMCORE
- Fix crash in add_astrometry if guider frame ra,dec not present (early SV data)
- Fix rare failure in fiber model parameter measurements
- Fix failure due to case change in Gaia query results
- Fix race condition due to non-unique filenames in astroquery (fixed upstream)
- Fix NaNs in SCISEN* header keywords
- Filter out QAFLAG BAD exposures and do not reduce
- Catch Error on very early guider frames with no RA, DEC keywords
- Add products documentation in header metadata with comments and physical units
- Improve fiber/wavelength thermal shift measurements in low SNR data
- Some other minor fixes and improvements

1.1.0 (30-10-2024)
------------------

- Implemented flux calibration using IFU field stars
- Improved twilight flat reduction and treatment for much improved fiberflats
- Add heliocentric velocity to headers
- More accurate resampling/rebinning code
- Improved wavelenth solution (especially in b channel) and more robust fits
- Allowing wavelength thermal shifts to vary along slit and improving wavelength/LSF fitting routines
- More robust and accurate fiber thermal shifts
- Add memory usage and execution time reporting
- Speed up pipeline across the board
- Adds a lockfile to drpall write, with 5 sec timeout, to prevent collisions during parallel writes.
- Adds `OBJECT` and `OBSTIME` header keys to the drpall summary file
- Added sky QA plots
- Improved QA plots across different routines
- Log exceptions to header COMMENT section
- Implement infrastructure for versioning for master calibration files
- Separate 2d reductions, extraction & 1d reduction, post-extraction stages, allow
  to run each individually
- More flexible CLI for cluster job submission
- Many bugfixes & stability/robustness improvements

1.0.3 (29-05-2024)
------------------

- Fixed mask propagation and RSS set to NaN by default in masked pixels
- Speeded up extraction
- Fixed Straylight bug manifesting in low SNR frames
- Fixed sky attributes propagation during channel combination

1.0.2 (28-05-2024)
------------------
- Speed up tracing by skipping model evaluation
- Fixes bug with missing astrometry headers
- Some other minor bug fixes

1.0.1 (23-05-2024)
------------------
- Fixed minor bugs in thermal shifts reports, sky QA plots, fetching of metadata, drpall product and more
- Implemented cleaning of ancillary paths
- Serial runs are default now

1.0.0 (22-05-2024)
------------------
- Retired the quick-reduction now all science reductions use drp run
- Integrated the calibration reductions with the drp
- Use of twilight flats to flatfield
- Remove the illumination gradient across the IFU in twilight exposures
- Detect and correct for the pixel shifts in raw frames
- Adjust the wavelength solutions to the sky lines in each frame
- Shift the traces to account for thermal motions in the spectrographs
- Implemented cosmic ray rejection
- Better pixel flats and bad pixel masks
- Subtract the scattered light for all reductions
- Better sky subtraction by separating sky continuum from sky lines
- Astrometry in the slitmap for each fiber, also guider astrometry in headers
- Have smaller calibration frames
- Provide new products: lvmFrame, lvmFFrame, lvmCFrame, lvmSFrame, lvmRSS
- A lot of bug fixes and small improvements

0.1.1 (11-08-2023)
------------------
- Tag of drp after new quick_reduction DRP, before changes for Utah

0.1.0 (07-13-2023)
------------------
- Initial tag for the current state of the DRP
