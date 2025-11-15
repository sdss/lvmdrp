.. _drp-changelog:

==========
Change Log
==========

This document records the main changes to the drp code.

1.2.0 (12-11-2025)
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
