Notes for Developers
====================

Notes for people who change the DRP code or prepare its releases. How to set up a
development installation and open a pull request is described in :doc:`Installation`.

Versions
--------

Two version strings
^^^^^^^^^^^^^^^^^^^

The DRP has two version strings:

- **The code version**: the ``version`` line in ``setup.cfg``. While a version is being
  developed it ends in ``dev`` (``2.0dev`` for the version after 1.3.2); a release drops
  the ``dev`` (``1.3.2``).
- **The version label**: ``$LVMDRP_VERSION`` if it is set, otherwise the code version. It is
  ``lvmdrp.__version__``::

      python -c "import lvmdrp; print(lvmdrp.__version__)"

The version label, not the code version, decides where the DRP writes:

- the reductions, under ``$LVM_SPECTRO_REDUX/<label>/<tilegrp>/<tileid>/<mjd>/``;
- the frames metadata (``raw_metadata.hdf5`` for each night, ``master_metadata.hdf5``)
  under ``$LVM_SPECTRO_REDUX/<label>/``;
- the drpall, ``$LVM_SPECTRO_REDUX/<label>/drpall-<label>.fits`` (and ``.h5``).

It is also written into the products as the header keyword ``DRPVER``. The master
calibrations are read from ``$LVM_MASTER_DIR`` and do not depend on the version.

When the code version changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``$LVMDRP_VERSION`` is not set, a change of the code version (a release, or a new
development version in ``setup.cfg``) changes the version label too, without any other
warning. After pulling the new code:

- new reductions go to a new directory, ``redux/<new label>``: for example ``redux/2.0dev``
  for the development code after 1.3.2, where the 1.3.2 release wrote to ``redux/1.3.2``;
- the metadata of each night is built again there the first time it is needed;
- earlier reductions, their metadata and their drpall stay in the old directory. The DRP
  does not look there, so ``--from-frame`` and similar options don't find earlier products,
  and the new drpall contains only what is reduced with the new label.

So, when the code version changes:

1. Decide where the reductions should go. For test reductions and long runs, set
   ``$LVMDRP_VERSION`` explicitly, so that a version change does not move them.
2. Check the label and the output directory before reducing anything, with ``envcheck``
   (see below).
3. Check anything that runs the DRP automatically, such as the daily run at Utah: it
   uses the label of the code it runs.

Checking the version with envcheck
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``envcheck`` shows both version strings and the directory the reductions will go to, in its
``outputs`` section. With the development code after 1.3.2 and ``$LVMDRP_VERSION`` not
set::

    [INFO]: outputs
    [INFO]:   version label          2.0dev  (the code version; LVMDRP_VERSION not set; names the redux directory)
    [INFO]:   code version           2.0dev  (setup.cfg; the label only when nothing else sets it)
    [INFO]:   reductions go to       <LVM_SPECTRO_REDUX>/2.0dev  (does not exist yet, can be created)

With the 1.3.2 release, the same lines show ``1.3.2`` and ``<LVM_SPECTRO_REDUX>/1.3.2``. With
``export LVMDRP_VERSION=my_tests`` the label and the directory become ``my_tests``, whatever
the code version::

    [INFO]:   version label          my_tests  (from LVMDRP_VERSION; names the redux directory)
    [INFO]:   code version           2.0dev  (setup.cfg; the label only when nothing else sets it)
    [INFO]:   reductions go to       <LVM_SPECTRO_REDUX>/my_tests  (exists, writable)

If the label or the directory is not what you expect, set ``$LVMDRP_VERSION`` before
reducing anything.

Releases
^^^^^^^^

Releases so far (for example 1.3.2, by Alfredo Mejia-Narvaez) have been made in two
commits:

1. "version bump": ``setup.cfg`` gets the release number (``1.3.2``), and the section
   ``<version>dev (unreleased)`` at the top of ``CHANGELOG.rst`` becomes the release
   section, ``<version> (<date>)``, with the changes of the release written into it. This
   commit is tagged with the release number.
2. "version bump -- new development cycle": ``setup.cfg`` gets the next development version
   (after 1.3.2 this was ``1.3.3dev``; it became ``2.0dev`` once the next release was
   planned as 2.0), and ``CHANGELOG.rst`` gets an empty ``<next version>dev (unreleased)``
   section at the top.

Pull requests don't edit ``CHANGELOG.rst``; it is written for the release.
