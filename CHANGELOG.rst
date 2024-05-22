.. _drp-changelog:

==========
Change Log
==========

This document records the main changes to the drp code.

1.0.0 (22-05-2024)
------------------

* Retired the quick-reduction now all science reductions use drp run
* Integrated the calibration reductions with the drp
* Use of twilight flats to flatfield
* Remove the illumination gradient across the IFU in twilight exposures
* Detect and correct for the pixel shifts in raw frames
* Adjust the wavelength solutions to the sky lines in each frame
* Shift the traces to account for thermal motions in the spectrographs
* Implemented cosmic ray rejection
* Better pixel flats and bad pixel masks
* Subtract the scattered light for all reductions
* Better sky subtraction by separating sky continuum from sky lines
* Astrometry in the slitmap for each fiber, also guider astrometry in headers
* Have smaller calibration frames
* Provide new products: lvmFrame, lvmFFrame, lvmCFrame, lvmSFrame, lvmRSS
* A lot of bug fixes and small improvements

0.1.2 (unreleased)
------------------


0.1.1 (11-08-2023)
------------------
- Tag of drp after new quick_reduction DRP, before changes for Utah

0.1.0 (07-13-2023)
------------------
- Initial tag for the current state of the DRP
