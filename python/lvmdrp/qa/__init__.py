# encoding: utf-8
"""Quality assessment of the LVM data reduction.

- :mod:`lvmdrp.qa.report`: shared theme, plot styling and HTML page used by the QA reports
- :mod:`lvmdrp.qa.fiberflats`: fiber flat fielding QA in flat-fielded science frames, and master fiber flats compared across the calibration epochs
- :mod:`lvmdrp.qa.pixelflats`: pixel flat sequences and products QA dashboards
- :mod:`lvmdrp.qa.science`: per-exposure QA dashboard of a science reduction, step by step
- :mod:`lvmdrp.qa.wavelengths`: master wavelength and LSF solutions of an epoch, from the lines of each arc exposure
- :mod:`lvmdrp.qa.arc_sky`: arc LSF of an epoch against the widths of sky lines in a science exposure
"""
