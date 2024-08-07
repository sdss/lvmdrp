.. _drp-changelog:

==========
Change Log
==========

This document records the main changes to the drp code.

1.1.0 (unreleased)
------------------

- Adds a lockfile to drpall write, with 5 sec timeout, to prevent collisions during parallel writes.
- Adds `OBJECT` and `OBSTIME` header keys to the drpall summary file
- Implemented flux calibration using IFU field stars
- Allowing wavelength thermal shifts to vary along slit and improving wavelength/LSF fitting routines
- Fixing fiber thermal shifts
- Speed up tracing
- Improved QA plots across different routines

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
