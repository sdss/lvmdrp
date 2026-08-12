Directory Structure
===================

This document describes the directory structure used by the LVM Data Reduction
Pipeline (DRP), including both input data locations and output product organization.

Environment Variables
---------------------

The DRP uses several environment variables to locate data and store outputs.
These are set automatically based on the SDSS tree configuration.

.. list-table:: Core Environment Variables
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``SAS_BASE_DIR``
     - Root directory of the Science Archive Server (SAS) mirror
   * - ``LVMCORE_DIR``
     - Path to the lvmcore repository containing configuration files
   * - ``LVM_DATA_S``
     - Raw data directory for LCO (South): ``$SAS_BASE_DIR/sdsswork/data/lvm/lco``
   * - ``LVM_SPECTRO_REDUX``
     - Reduction output directory: ``$SAS_BASE_DIR/sdsswork/lvm/spectro/redux``
   * - ``LVM_MASTER_DIR``
     - Master calibrations: ``$SAS_BASE_DIR/sdsswork/lvm/sandbox/calib``
   * - ``LVM_SANDBOX``
     - Sandbox directory: ``$SAS_BASE_DIR/sdsswork/lvm/sandbox``

Input Data Structure
--------------------

Raw Data
^^^^^^^^

Raw science and calibration frames are stored in the ``LVM_DATA_S`` directory,
organized by MJD:

.. code-block:: text

    $LVM_DATA_S/
    └── <mjd>/
        ├── sdR-s-b1-<expnum>.fits.gz
        ├── sdR-s-b2-<expnum>.fits.gz
        ├── sdR-s-b3-<expnum>.fits.gz
        ├── sdR-s-r1-<expnum>.fits.gz
        ├── sdR-s-r2-<expnum>.fits.gz
        ├── sdR-s-r3-<expnum>.fits.gz
        ├── sdR-s-z1-<expnum>.fits.gz
        ├── sdR-s-z2-<expnum>.fits.gz
        └── sdR-s-z3-<expnum>.fits.gz

Raw frame naming convention:

- ``sdR`` - SDSS Raw frame prefix
- ``s`` - Hemisphere (s = South/LCO)
- ``b1``, ``r2``, ``z3``, etc. - Camera (channel + spectrograph number)
- ``<expnum>`` - 8-digit exposure number (e.g., ``00012345``)

Master Calibrations
^^^^^^^^^^^^^^^^^^^

Long-term master calibration files are stored in the sandbox, organized by
calibration epoch (MJD):

.. code-block:: text

    $LVM_MASTER_DIR/
    ├── <mjd>/                          # Calibration epoch
    │   ├── lvm-mbias-b1.fits           # Master bias
    │   ├── lvm-mbias-b2.fits
    │   ├── ...
    │   ├── lvm-mcentroids-b1.fits      # Fiber trace centroids
    │   ├── lvm-msigmas-b1.fits         # Fiber trace widths
    │   ├── lvm-mmodel-b1.fits          # Fiber PSF model
    │   ├── lvm-mwave-b1.fits           # Wavelength solution
    │   ├── lvm-mlsf-b1.fits            # Line spread function
    │   ├── lvm-mfiberflat_twilight-b.fits  # Twilight fiberflat (per channel)
    │   └── ...
    ├── pixelmasks/                     # Pixel masks (version-independent)
    │   ├── lvm-mpixmask-b1.fits
    │   ├── lvm-mpixflat-b1.fits        # Pixel flat
    │   └── ...
    ├── gaia_cache/                     # Cached Gaia XP spectra for flux calibration
    └── stellar_models/                 # Stellar atmosphere templates

Calibration file naming convention:

- ``lvm-m<type>-<camera>.fits`` - Master calibration per camera
- ``lvm-m<type>-<channel>.fits`` - Master calibration per channel (b, r, z)

LVM Core
^^^^^^^^

The lvmcore repository contains essential configuration files:

.. code-block:: text

    $LVMCORE_DIR/
    ├── etc/
    │   ├── exclude_mjds.txt            # MJDs to skip during reduction
    │   ├── pixel_shifts.parquet        # Pixel shift corrections
    │   └── ...
    └── pixelshifts/                    # Validated electronic pixel shifts

Output Data Structure
---------------------

Reduction outputs are organized under ``LVM_SPECTRO_REDUX`` with a hierarchical
structure based on DRP version, tile group, tile ID, and MJD.

Overview
^^^^^^^^

.. code-block:: text

    $LVM_SPECTRO_REDUX/
    └── <drpver>/                       # DRP version (e.g., "1.0.0" or "my_tests")
        ├── lvm-config-<drpver>.yaml    # DRP configuration snapshot
        ├── master_metadata.hdf5        # Master calibration metadata cache
        ├── drpall-<drpver>.h5          # Summary file (HDF5)
        ├── drpall-<drpver>.fits        # Summary file (FITS)
        └── <tilegrp>/                  # Tile group (e.g., "0011XX")
            └── <tileid>/               # Tile ID (e.g., "11111")
                ├── pixelmasks/         # Version-specific pixel masks (optional)
                └── <mjd>/              # MJD of observation
                    ├── raw_metadata.hdf5
                    ├── lvm-drp-<tileid>-<mjd>.log
                    ├── ancillary/      # Intermediate products
                    ├── lvmFrame-*.fits # Extracted frames
                    ├── lvmFFrame-*.fits # Flux-calibrated frames
                    ├── lvmCFrame-*.fits # Channel-combined frames
                    └── lvmSFrame-*.fits # Sky-subtracted frames

Tile Groups
^^^^^^^^^^^

Tile IDs are organized into tile groups to limit the number of subdirectories.
The tile group is derived from the tile ID:

- Tile ID ``11111`` → Tile group ``0011XX``
- Tile ID ``12345`` → Tile group ``0012XX``

This is computed by taking the first 4 digits and appending ``XX``.

Ancillary Products
^^^^^^^^^^^^^^^^^^

Intermediate reduction products are stored in the ``ancillary/`` subdirectory:

.. code-block:: text

    ancillary/
    ├── lvm-pobject-<camera>-<expnum>.fits    # Preprocessed
    ├── lvm-dobject-<camera>-<expnum>.fits    # Detrended
    ├── lvm-dstray-<camera>-<expnum>.fits     # Stray light model
    ├── lvm-lobject-<camera>-<expnum>.fits    # Stray light subtracted
    ├── lvm-xobject-<camera>-<expnum>.fits    # Extracted (per camera)
    ├── lvm-xobject-<channel>-<expnum>.fits   # Extracted (stacked per channel)
    ├── lvm-wobject-<channel>-<expnum>.fits   # Wavelength calibrated
    ├── lvm-sobject-<channel>-<expnum>.fits   # Sky interpolated
    ├── lvm-hobject-<channel>-<expnum>.fits   # Wavelength resampled
    ├── lvm-wsky_e-<channel>-<expnum>.fits    # East sky telescope
    ├── lvm-wsky_w-<channel>-<expnum>.fits    # West sky telescope
    └── qa/                                    # Quality assurance plots

Ancillary file prefix codes:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Prefix
     - Description
   * - ``p``
     - Preprocessed (overscan subtracted, pixel masked)
   * - ``d``
     - Detrended (bias/dark subtracted, pixel flat corrected)
   * - ``l``
     - Stray Light subtracted
   * - ``x``
     - Extracted (1D fiber spectra)
   * - ``w``
     - Wavelength calibrated
   * - ``s``
     - Sky interpolated
   * - ``h``
     - Resampled to uniform wavelength grid

Final Data Products
^^^^^^^^^^^^^^^^^^^

The main science products are stored at the MJD level:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File Pattern
     - Description
   * - ``lvmFrame-<channel>-<expnum>.fits``
     - Extracted, wavelength calibrated, flatfielded (in electrons)
   * - ``lvmFFrame-<channel>-<expnum>.fits``
     - Flux calibrated (in physical units)
   * - ``lvmCFrame-<expnum>.fits``
     - Channel combined (full wavelength coverage)
   * - ``lvmSFrame-<expnum>.fits``
     - Sky subtracted (final science product)

Where:

- ``<channel>`` is ``b``, ``r``, or ``z``
- ``<expnum>`` is the 8-digit exposure number

Guide Camera Products
^^^^^^^^^^^^^^^^^^^^^

Guide camera coadds are stored separately and used for astrometry:

.. code-block:: text

    $SAS_BASE_DIR/sdsswork/data/lvm/lco/<mjd>/coadds/
    ├── lvm.sci.coadd_s<expnum>.fits    # Science telescope
    ├── lvm.skye.coadd_s<expnum>.fits   # Sky-East telescope
    └── lvm.skyw.coadd_s<expnum>.fits   # Sky-West telescope

Camera and Channel Nomenclature
-------------------------------

The LVM instrument has three spectrographs (sp1, sp2, sp3), each with three
spectral channels (b, r, z):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Camera
     - Channel
     - Wavelength Range (Angstrom)
   * - b1, b2, b3
     - Blue (b)
     - 3600 - 5800
   * - r1, r2, r3
     - Red (r)
     - 5775 - 7570
   * - z1, z2, z3
     - NIR (z)
     - 7520 - 9800

The number in the camera name indicates the spectrograph (1, 2, or 3).

- **Camera-level files** (e.g., ``b1``, ``r2``): Individual CCD data
- **Channel-level files** (e.g., ``b``, ``r``, ``z``): Combined from all three spectrographs

Fiber Organization
------------------

Each spectrograph has 648 fibers organized into 18 blocks of 36 fibers each.
The full instrument has 1944 fibers total (648 x 3 spectrographs).

Fibers are allocated to different telescopes:

- **Sci** - Science IFU fibers
- **SkyE** - Sky-East telescope fibers
- **SkyW** - Sky-West telescope fibers
- **Spec** - Standard star fibers

The fiber mapping is stored in the ``SLITMAP`` extension of RSS files and
is defined in the lvmcore repository.

Example Directory Layout
------------------------

A complete example for MJD 60255, tile 11111, DRP version 1.0.0:

.. code-block:: text

    $SAS_BASE_DIR/
    └── sdsswork/
        ├── data/lvm/lco/
        │   └── 60255/
        │       ├── sdR-s-b1-00060255.fits.gz
        │       ├── sdR-s-r1-00060255.fits.gz
        │       └── ...
        └── lvm/
            ├── sandbox/
            │   └── calib/
            │       ├── 60200/              # Calibration epoch
            │       │   ├── lvm-mbias-b1.fits
            │       │   └── ...
            │       └── pixelmasks/
            │           └── lvm-mpixmask-b1.fits
            └── spectro/
                └── redux/
                    └── 1.0.0/
                        ├── drpall-1.0.0.fits
                        └── 0011XX/
                            └── 11111/
                                └── 60255/
                                    ├── raw_metadata.hdf5
                                    ├── ancillary/
                                    │   ├── lvm-pobject-b1-00060255.fits
                                    │   └── ...
                                    ├── lvmFrame-b-00060255.fits
                                    ├── lvmFFrame-b-00060255.fits
                                    ├── lvmCFrame-00060255.fits
                                    └── lvmSFrame-00060255.fits
