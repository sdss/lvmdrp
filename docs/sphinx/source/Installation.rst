Installation
============

This guide describes how to install and configure the LVM Data Reduction Pipeline (DRP).

The LVM DRP is based on a collection of routines from `Py3D <https://github.com/brandherd/Py3D>`_.

Prerequisites
-------------

The DRP is developed and tested on Ubuntu-based systems using **Python 3.10**. We recommend
using a Python environment manager such as Anaconda to maintain a consistent Python version
and avoid conflicts with the system Python installation.

We assume you are a member of the `Github SDSS organization <https://github.com/sdss>`_ and
have an SSH key configured for your Github account. If not, please follow
`these instructions <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_
to set one up.

1. Clone and Configure LVM Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the current version of `LVM Core <https://github.com/sdss/lvmcore>`_:

.. code-block:: bash

    git clone git@github.com:sdss/lvmcore.git

Set the environment variable ``LVMCORE_DIR`` pointing to the root directory in your
``.bashrc`` (or equivalent):

.. code-block:: bash

    export LVMCORE_DIR="path/to/lvmcore"

2. Configure SAS Base Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the ``SAS_BASE_DIR`` environment variable in your ``.bashrc`` (or equivalent) to
point to your local mirror of the SAS:

.. code-block:: bash

    export SAS_BASE_DIR="path/to/sas-root-directory"

You can download a target MJD from the SAS while preserving the directory structure:

.. code-block:: bash

    wget -X css --reject html -nH -nc -t0 -r --level=2 -E --ignore-length -x -k -p -erobots=off -np -N \
        https://data.sdss5.org/sas/sdsswork/data/lvm/lco/<mjd>/ \
        --user <user> --password <password>

.. note::

    We strongly recommend using the `sdss_access <https://github.com/sdss/sdss_access>`_
    product instead of wget for downloading data.

3. Create Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new Python environment (optional but strongly recommended):

.. code-block:: bash

    conda create -n lvmdrp python=3.10

Activate the environment:

.. code-block:: bash

    conda activate lvmdrp

Installing the DRP
------------------

If you are installing on a system other than Ubuntu (e.g., macOS), please read the
:ref:`troubleshooting` section before proceeding.

1. Clone the repository:

   .. code-block:: bash

       git clone git@github.com:sdss/lvmdrp.git

2. Change to the ``lvmdrp`` directory:

   .. code-block:: bash

       cd lvmdrp

3. Install the DRP package:

   .. code-block:: bash

       pip install .

   For development installation, see the :ref:`contributing` section below.

Testing the Installation
------------------------

Use the ``envcheck`` tool to verify that all required environment variables are set:

.. code-block:: bash

    envcheck

If the variables are correctly set, you should see the values of each variable and a
success message.

Setting Up Calibration Files
----------------------------

Download the current set of calibrations from the
`SAS sandbox <https://data.sdss5.org/sas/sdsswork/lvm/sandbox/calib/>`_.

After installation, you can use the ``drp get-calibs`` command. For usage information:

.. code-block:: bash

    drp get-calibs --help

To download all calibration files for a specific MJD (e.g., 60255):

.. code-block:: bash

    drp get-calibs -m 60255

This command downloads the files using ``sdss-access`` and places them in ``$LVM_MASTER_DIR``,
which is defined by the pipeline as ``$LVM_SANDBOX/calib``, mirroring the SAS structure.
The files will be located at ``$SAS_BASE_DIR/sdsswork/lvm/sandbox/calib/``.

To download all calibration epochs:

.. code-block:: bash

    drp get-calibs

.. note::

    When choosing which calibration epoch to use for a specific exposure, always select
    the epoch that is closest to and earlier than your exposure's MJD.

Required Environment Variables
------------------------------

The following environment variables must be set:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``LVMCORE_DIR``
     - Path to the lvmcore repository
   * - ``SAS_BASE_DIR``
     - Path to the SAS root directory

The following variables are set automatically by the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``LVM_MASTER_DIR``
     - Auto-set to ``$SAS_BASE_DIR/sdsswork/lvm/sandbox/calib/``

Optional environment variables:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``LVM_DRPVER``
     - DRP version string for output directory naming (useful for testing)
   * - ``LVM_ESOSKY_DIR``
     - Path to ESO sky routines (only needed for sky module)

ESO Sky Routines (Optional)
---------------------------

.. note::

    You do not need to install the ESO sky routines to run standard science reductions.

If you plan to use the sky module, you will need to install the ESO routines first.
To install `skycorr <https://www.eso.org/sci/software/pipelines/skytools/skycorr>`_ and
the `ESO Sky Model <https://www.eso.org/sci/software/pipelines/skytools/skymodel>`_,
follow the instructions at:

- https://wiki.sdss.org/display/LVM/ESO+skycorr
- https://wiki.sdss.org/display/LVM/ESO+Sky+Model

After installation, set the following environment variable in your ``.bashrc``:

.. code-block:: bash

    export LVM_ESOSKY_DIR="path/to/eso-routines"

where ``eso-routines`` is a directory containing the root directories of both the
*skycorr* and the *ESO sky model* installations.

.. _contributing:

Contributing to Development
---------------------------

There are two ways to contribute:

- Testing the DRP and reporting bugs on Github
- Diving into the code to fix bugs and implement new features

For those contributing code, follow these steps:

1. Install the pipeline in editable (developer) mode:

   .. code-block:: bash

       pip install -e '.[dev]'

2. Before coding a new feature or bug fix, ensure your local ``master`` branch is up to date:

   .. code-block:: bash

       git pull origin master

3. Create a branch with a descriptive name:

   .. code-block:: bash

       git checkout -b <feature_name>

4. Make your changes. When ready to commit:

   .. code-block:: bash

       git status  # Check what has changed
       git add <changed_files>
       git commit -m "commit message"

5. Push your updates to the remote branch:

   .. code-block:: bash

       git push

6. Create a `pull request on Github <https://github.com/sdss/lvmdrp/pulls>`_ when ready to merge.

7. To keep different DRP output versions in separate directories, set:

   .. code-block:: bash

       export LVM_DRPVER="my_tests"

   This will store outputs in:

   .. code-block:: bash

       $SAS_BASE_DIR/sdsswork/lvm/spectro/redux/my_tests/<tilegrp>/<tileid>/<mjd>/

.. tip::

    Use an *atomic* commit approach where each commit has a single purpose. Avoid pushing
    lots of unrelated changes in one commit.

.. _troubleshooting:

Troubleshooting
---------------

macOS users may need to perform extra steps before the standard installation.

Issue importing CSafeLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some Mac users encounter this error when importing from PyYAML (~6.0):

.. code-block:: text

    AttributeError: module 'yaml' has no attribute 'CSafeLoader'

PyYAML is installed as a dependency of PyTables. The problem can be solved by either:

- Installing PyTables from conda directly (instead of pip)
- Installing PyTables from their `master branch <https://github.com/PyTables/PyTables>`_

macOS Monterey (v12.6.2)
^^^^^^^^^^^^^^^^^^^^^^^^

Run this extra step before the DRP installation:

.. code-block:: bash

    sudo port install py38-healpy

See the `healpy documentation <https://healpy.readthedocs.io/en/latest/install.html#compilation-issues-with-mac-os>`_
for more information on this issue.

macOS Mojave (v10.14.6)
^^^^^^^^^^^^^^^^^^^^^^^

The installation of ``scipy`` (a core DRP dependency) requires OpenBLAS for compilation.
On older macOS versions:

1. Install OpenBLAS:

   .. code-block:: bash

       brew install openblas

2. Set ``$PKG_CONFIG_PATH`` to point to your OpenBLAS installation:

   .. code-block:: bash

       export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"

After these steps, proceed with the standard DRP installation.
