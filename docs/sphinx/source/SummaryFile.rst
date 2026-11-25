DRPall Summary File
====================

This document describes the ``drpall`` summary file: what it is, how it is
built, and — most usefully — exactly which header keyword in the
``lvmSFrame`` files populates each column.

Overview
--------

The ``drpall`` file is a single row-per-exposure summary table, not a
collection of per-exposure tables. The output FITS file
(``drpall-<drpver>.fits``) has two HDUs:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - HDU
     - Name
     - Content
   * - 0
     - ``PRIMARY``
     - Empty primary header
   * - 1
     - ``LVM``
     - One row per science exposure, one column per summarized quantity

It is built by :func:`~lvmdrp.utils.metadata.update_summary_file`
(``python/lvmdrp/utils/metadata.py``), which is called once per exposure at
the end of :func:`~lvmdrp.main.science_reduction` (see the
:ref:`Summary Generation <summary-generation>` stage of the pipeline
overview). For each exposure it:

1. Opens the ``PRIMARY`` header of that exposure's ``lvmSFrame-<expnum>.fits``
   file and extracts a fixed set of keywords via the internal helper
   ``_collect_header_data``.
2. Combines those values with a handful of columns taken from the raw
   exposure metadata database (not the header) and a few paths/values
   computed on the spot.
3. Appends the resulting row to a per-``drpver`` HDF5 file
   (``drpall-<drpver>.h5``), keyed on ``expnum``, via
   :func:`~lvmdrp.utils.metadata.safe_hdf_append`.

Once a whole DRP run has completed, :func:`~lvmdrp.main.create_drpall`
converts the accumulated HDF5 table to the final FITS file via
:func:`~lvmdrp.utils.metadata.convert_h5_to_fits`.

Column-to-Header-Keyword Mapping
---------------------------------

Unless noted otherwise, all keywords below are read from the ``PRIMARY``
header of the corresponding exposure's ``lvmSFrame`` file. Keywords written
as ``HIERARCH ...`` are long/compound FITS keywords (containing a space) and
are looked up with ``header.get("SKY SCI_ALT")`` etc.

Identification and DRP Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``drpver``
     - *(not from header)*
     - Set from the ``drpver`` argument passed to
       :func:`~lvmdrp.utils.metadata.update_summary_file`. The value is also
       initially read from ``DRPVER``, but is overwritten before the row is
       assembled — the final column reflects the requested reduction run, not
       necessarily the ``DRPVER`` stamped in that file's header.
   * - ``drpqual``
     - ``DRPQUAL``
     - Overwrites a same-named column already present from the raw exposure
       metadata (see :ref:`summary-non-header-columns`); the header value wins.
   * - ``dpos``
     - ``DPOS``
     -
   * - ``object``
     - ``OBJECT``
     - Empty values are filled with the string ``"None"``.
   * - ``obstime``
     - ``OBSTIME``
     -

Flux Calibration
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``std_mean_senb`` / ``std_mean_senr`` / ``std_mean_senz``
     - ``STDSENMB`` / ``STDSENMR`` / ``STDSENMZ``
     - Mean sensitivity from standard stars, per channel
   * - ``sci_mean_senb`` / ``sci_mean_senr`` / ``sci_mean_senz``
     - ``SCISENMB`` / ``SCISENMR`` / ``SCISENMZ``
     - Mean sensitivity from science-IFU field stars, per channel
   * - ``mod_mean_senb`` / ``mod_mean_senr`` / ``mod_mean_senz``
     - ``MODSENMB`` / ``MODSENMR`` / ``MODSENMZ``
     - Mean sensitivity from the model sensitivity curve, per channel
   * - ``std_rms_senb`` / ``std_rms_senr`` / ``std_rms_senz``
     - ``STDSENRB`` / ``STDSENRR`` / ``STDSENRZ``
     - RMS sensitivity from standard stars, per channel
   * - ``sci_rms_senb`` / ``sci_rms_senr`` / ``sci_rms_senz``
     - ``SCISENRB`` / ``SCISENRR`` / ``SCISENRZ``
     - RMS sensitivity from science-IFU field stars, per channel
   * - ``mod_rms_senb`` / ``mod_rms_senr`` / ``mod_rms_senz``
     - ``MODSENRB`` / ``MODSENRR`` / ``MODSENRZ``
     - RMS sensitivity from the model sensitivity curve, per channel
   * - ``fluxcal``
     - ``FLUXCAL``
     - Flux calibration method/status flag

Science Telescope (``sci``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``sci_ra`` / ``sci_dec``
     - ``SCIRA`` / ``SCIDEC``
     -
   * - ``sci_pa``
     - ``SCIPA``
     -
   * - ``sci_amass``
     - ``SCIAM``
     - Airmass
   * - ``sci_astsrc``
     - ``SCIASRC``
     - Astrometric solution source
   * - ``sci_kmpos``
     - ``TESCIKM``
     - K-mirror position
   * - ``sci_focpos``
     - ``TESCIFO``
     - Focus stage position
   * - ``sci_alt``
     - ``HIERARCH SKY SCI_ALT``
     - Altitude above horizon [deg]
   * - ``sci_sh_hght``
     - ``HIERARCH SKY SCI_SH_HGHT``
     - Height of Earth's shadow [km]
   * - ``sci_moon_sep``
     - ``HIERARCH SKY SCI_MOON_SEP``
     - Separation from the Moon [deg]

East Sky Telescope (``skye``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``skye_ra`` / ``skye_dec``
     - ``SKYERA`` / ``SKYEDEC``
     -
   * - ``skye_pa``
     - ``SKYEPA``
     -
   * - ``skye_amass``
     - ``SKYEAM``
     - Airmass
   * - ``skye_astsrc``
     - ``SKYEASRC``
     - Astrometric solution source
   * - ``skye_kmpos``
     - ``TESKYEKM``
     - K-mirror position
   * - ``skye_focpos``
     - ``TESKYEFO``
     - Focus stage position
   * - ``skye_name``
     - ``SKYENAME``
     - Empty values are filled with the string ``"None"``.
   * - ``skye_alt``
     - ``HIERARCH SKY SKYE_ALT``
     - Altitude above horizon [deg]
   * - ``sci_skye_sep``
     - ``HIERARCH SKY SCI_SKYE_SEP``
     - Separation between Sci and SkyE pointings [deg]
   * - ``skye_sh_hght``
     - ``HIERARCH SKY SKYE_SH_HGHT``
     - Height of Earth's shadow [km]
   * - ``skye_moon_sep``
     - ``HIERARCH SKY SKYE_MOON_SEP``
     - Separation from the Moon [deg]

West Sky Telescope (``skyw``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``skyw_ra`` / ``skyw_dec``
     - ``SKYWRA`` / ``SKYWDEC``
     -
   * - ``skyw_pa``
     - ``SKYWPA``
     -
   * - ``skyw_amass``
     - ``SKYWAM``
     - Airmass
   * - ``skyw_astsrc``
     - ``SKYWASRC``
     - Astrometric solution source
   * - ``skyw_kmpos``
     - ``TESKYWKM``
     - K-mirror position
   * - ``skyw_focpos``
     - ``TESKYWFO``
     - Focus stage position
   * - ``skyw_name``
     - ``SKYWNAME``
     - Empty values are filled with the string ``"None"``.
   * - ``skyw_alt``
     - ``HIERARCH SKY SKYW_ALT``
     - Altitude above horizon [deg]
   * - ``sci_skyw_sep``
     - ``HIERARCH SKY SCI_SKYW_SEP``
     - Separation between Sci and SkyW pointings [deg]
   * - ``skyw_sh_hght``
     - ``HIERARCH SKY SKYW_SH_HGHT``
     - Height of Earth's shadow [km]
   * - ``skyw_moon_sep``
     - ``HIERARCH SKY SKYW_MOON_SEP``
     - Separation from the Moon [deg]

Sun / Moon Geometry
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``drpall`` column
     - Header keyword
     - Notes
   * - ``moon_ra`` / ``moon_dec``
     - ``HIERARCH SKY MOON_RA`` / ``HIERARCH SKY MOON_DEC``
     -
   * - ``moon_phase``
     - ``HIERARCH SKY MOON_PHASE``
     - 0=new, 90=1st quarter, 180=full, 270=3rd quarter [deg]
   * - ``moon_fli``
     - ``HIERARCH SKY MOON_FLI``
     - Fraction of lunar illumination
   * - ``moon_alt``
     - ``HIERARCH SKY MOON_ALT``
     - Altitude above horizon [deg]
   * - ``sun_alt``
     - ``HIERARCH SKY SUN_ALT``
     - Altitude above horizon [deg]

All ``HIERARCH SKY *`` values are themselves computed and written into the
``lvmSFrame`` header earlier in the pipeline by
:func:`~lvmdrp.core.sky.sky_pars_header`, which is the ultimate source if
tracing these further back is needed.

.. _summary-non-header-columns:

Columns Not Sourced From the Header
-------------------------------------

A few ``drpall`` columns are not read from the ``lvmSFrame`` header at all:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``drpall`` column
     - Source
   * - ``tilegrp``, ``tileid``, ``mjd``, ``expnum``, ``exptime``, ``stage``, ``status``
     - The raw exposure metadata database, via
       :func:`~lvmdrp.utils.metadata.get_metadata` — not the ``lvmSFrame``
       header.
   * - ``filename``, ``location``, ``agcam_location``
     - Constructed from ``sdss_access`` path templates for the given
       ``tileid``/``mjd``/``expnum``/``drpver``.
   * - ``calib_mjd``
     - Passed in directly as the ``master_mjd`` argument to
       :func:`~lvmdrp.utils.metadata.update_summary_file`.

Note that ``drpqual`` is present in both sources: the metadata database
supplies an initial value as part of the ``row`` selection, but it is then
overwritten by the header's ``DRPQUAL`` keyword when the two are merged, so
the header value is what ends up in the final table.

Data Product Blueprint
------------------------

The formal blueprint for this product,
``python/lvmdrp/etc/dataproducts/lvmDrpAll_bp.yaml``, is currently a stub — it
only documents the ``TILEID`` column and a placeholder description. The
authoritative column list is the one produced by the code and documented
above; the blueprint has not been kept in sync with it.
