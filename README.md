# Local Volume Mapper (LVM) Data Reduction Pipeline (DRP)

The LVM DRP is based in a collection of routines from [Py3D](https://github.com/brandherd/Py3D).

## Prerequisites

To properly run the DRP you need to prepare your environment by following these steps:

1. Download the current version of [LVM Core](https://github.com/sdss/lvmcore):

    ```bash
    git clone git@github.com:sdss/lvmcore.git     # assuming you have an SSH key
    ```
    or
    ```bash
    git clone https://github.com/sdss/lvmcore.git # using HTTPS
    ```

    and set the environment variable `LVM_CORE_DIR` pointing to the root directory `lvmcore` in your `.bashrc` (or equivalent):

    ```bash
    export LVM_CORE_DIR="path/to/lvmcore"
    ```

2. Define the environment variable for your mirror of the SAS:

    ```bash
    export SAS_BASE_DIR="path/to/sas-root-directory"
    ```


## Installation

This code is being developed/tested in a Ubuntu-based OS, using **Python 3.8**. We recommend to use a Python environment manager such as Anaconda or similar, in order to avoid cluttering the OS's python installation. We assume you are a member of the [Github sdss organization](https://github.com/sdss). We also assume that you have an SSH key configure on your local machine and registered in your Github account. If not, please follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to set up one.

If you are planning on installing the DRP on a different OS, please read the [troubleshooting section](#troubleshooting) before you continue with the steps below.

To install the DRP along with its dependencies, you need to run the following steps:

1. Create a **Python 3.8** environment. This is optional, but strongly recommended. With conda this is done like this:
   
    ```bash
    conda create -n lvmdrp python=3.8
    ```

2. Make sure you are in the intended **Python 3.8** environment and directory:
    
    ```bash
    conda activate lvmdrp
    ```

3. Clone the Github repository:
    
    ```bash
    git clone git@github.com:sdss/lvmdrp.git     # assuming you have an SSH key
    ```
    or

    ```bash
    git clone https://github.com/sdss/lvmdrp.git # using HTTPS
    ```

4. Go into the `lvmdrp` directory:
    
    ```bash
    cd lvmdrp
    ```


5. Install the DRP package in the current python environment (see [contributing](#contributing-to-lvm-drp-development) section below for a replacement of this step):
    
    ```bash
    pip install .
    ```

6. Download the current set of calibrations from the [SAS sandbox](https://data.sdss5.org/sas/sdsswork/lvm/sandbox/calib/) and add to your `.bashrc` (or equivalent) the following definition:

    ```bash
    export LVM_MASTER_DIR="path/to/master-calibrations"
    ```

    where `master-calibrations` contains only MJD folders.

## Testing the installation

<!-- write a script to test everything finished correctly with the installation -->

## ESO sky routines installation and configuration

If you are planning on using the sky module, you will need to install the ESO routines first. To install [skycorr](https://www.eso.org/sci/software/pipelines/skytools/skycorr) and the [ESO Sky Model](https://www.eso.org/sci/software/pipelines/skytools/skymodel), follow the instructions in the following links:

- https://wiki.sdss.org/display/LVM/ESO+skycorr
- https://wiki.sdss.org/display/LVM/ESO+Sky+Model

respectively. Additionaly, you'll need to set following the environment variable on your `.bashrc` (or equivalent):

```bash
export LVM_ESOSKY_DIR="path/to/eso-routines"
```

where `eso-routines` is a directory containing the root directories of both, the `skycorr` and the ESO sky model installations.

## Creating Test Data

We encourage the reader to use the [LVM data simulator](https://github.com/sdss/lvmdatasimulator) to generate data for testing the DRP. But if you want to skip that step, we have already some simulations produced using the same simulator, so you don't have to run the simulator yourself, which can be computationally demanding in the case of 2D simulations.

If you follow the [examples](#examples) below, you will have access to the above mentioned simulations.

## Examples

You will find tutorial notebooks to run different DRP routines in the [examples](https://github.com/sdss/lvmdrp/tree/master/examples) folder. Here is you will find Jupyter Notebooks that illustrate different tasks in the DRP:

- [Basic Calibration](): reduction of calibration images: bias, dark and pixel flats; as well as reduction of arcs and fiber flats.
- [Wavelength Calibration](): automatic pixel to wavelength mapping and wavelength and LSF fitting.
- [Flux Calibration](): conversion of the extracted spectra from electrons to flux calibrated spectra.
- [Sky Module](https://github.com/sdss/lvmdrp/tree/master/examples/sky_module): several procedures to sky-subtract science spectra.

In each of the links above you will find a short description of what's going on in each example and also the order in which those are intended to be followed.

## Contributing to LVM-DRP development

There are two ways in which you can contribute:

- Testing the DRP and reporting bugs on Github or
- By diving into the code to fix bugs and implement new features

For those willing to contribute by coding, there are some steps to streamline the development process:

1. Make sure you install the pipeline on your environment in edit (developer) mode, like this:

    ```bash
    pip install -e .
    ```

2. Before you start coding on a new feature/bug-fix, make sure your **local** `master` branch is up to date:

    ```bash
    git pull master
    ```

3. Create a branch to work on and make sure the name can be easily mappable to the work you intend to do:

    ```bash
    git checkout -b <feature_name>
    ```

4. Afterwards, you can push your updates to the remote branch on Github:

    ```bash
    git push
    ```

5. Finally, if you consider your feature is ready to be merged to the `master` branch, you can create a new [pull request at Github](https://github.com/sdss/lvmdrp/pulls).

Regarding commits, I'm trying to go for an *atomic* approach, where each commit has a single purpose. So please try to avoid as much as possible pushing lots of unrelated changes in one commit.

## Troubleshooting

In some MacOS versions there may be the need to perform extra installation steps, before getting into the steps described in the [installation section](#installation).

### Issue importing CSafeLoader

Some Mac users have found the folloring error while importing `CSafeLoader` from the PyYaml package (~6.0):

```python
AttributeError: module 'yaml' has no attribute 'CSafeLoader'
```

PyYaml is being installed as a dependency of PyTable. As of **Aug 7, 2023**, the problem seems to be solved by either installing PyTables from conda directly (instead of using pip install) or by installing PyTables from their [master branch](https://github.com/PyTables/PyTables).

### For MacOS (Monterey v12.6.2)

You will require to run this extra step before continuing with the regular DRP installation:

```bash
sudo port install py38-healpy
```

See [healpy documentation](https://healpy.readthedocs.io/en/latest/install.html#compilation-issues-with-mac-os) for a statement on this issue.

After this step, you should be able to proceed with the DRP installation as described in the [installation section](#installation).

### For MacOS (Mojave v10.14.6)

The installation of the `scipy` package (a core dependency of the DRP) requires openBLAS to be installed to be able to compile the source files. If you are running on an old MacOS version, please follow these steps:

1. Install `openBLAS` by doing:

    ```bash
    brew install openblas
    ```

2. Set `$PKG_CONFIG_PATH` to point to your installation of `openBLAS`. This may look like this:

    ```bash
    export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"
    ```

After these steps, you should be able to proceed with the DRP installation as described in the [installation section](#installation).
