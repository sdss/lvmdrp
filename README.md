# Local Volume Mapper (LVM) Data Reduction Pipeline (DRP)

The LVM DRP in it's current incarnation installs a collection of routines which make use of the [Py3D]().

## Installation

This code is being developed/tested in a Ubuntu-based OS, using **Python 3.8**. We recommend to use a Python environment manager such as Anaconda or similar, in order to avoid cluttering the OS's python installation.

**NOTE**: MacOS (Monterrey v12.6.2) users, will require to run this extra step before continuing with the regular DRP installation:
> `sudo port install py38-healpy`

See [healpy documentation](https://healpy.readthedocs.io/en/latest/install.html#compilation-issues-with-mac-os) for a statement on this issue.

To install the DRP along with its dependencies, you need to run the following steps:

1. Create a **Python 3.8** environment. This is optional, but strongly recommended. With conda this is done like this:
    > `conda create -n lvmdrp python=3.8`

2. Make sure you are in the intended **Python 3.8** environment and directory:
    > `conda activate lvmdrp`

3. Clone the Github repository:
    > `git clone --recurse-submodules -j8 git@github.com:sdss/lvmdrp.git`

4. Go into the lvmdrp directory:
    > `cd lvmdrp`

5. Switch to the (current) development branch:
    > `git checkout development`

6. Install the DRP package in the current python environment (see *Contributing* section below for a replacement of this step):
    > `pip install --no-cache-dir . 2>&1 | tee today.txt`

## Testing the installation

<!-- write a script to test everything finished correctly with the installation -->

### Advanced ESO sky model configuration

If you are planning on using the sky module, you will need to install the ESO routines first. In order to install to do so
you need to run the following commands, **also within DRP python environment**.

1. Install the ESO skycorr and skymodel routines:
    > `drp sky installESOSky`

2. Run the sky module configuration:
    > `drp sky configureSkyModel`

## Tutorials

<!-- write tutorial notebooks -->
You will find tutorial notebooks to run different DRP routines in the [examples]() folder. Here is a list of the tutorials

## Creating Test Data
