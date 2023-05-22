# Local Volume Mapper (LVM) Data Reduction Pipeline (DRP)

The LVM DRP is based in a collection of routines from [Py3D](https://github.com/brandherd/Py3D).

## Installation

This code is being developed/tested in a Ubuntu-based OS, using **Python 3.8**. We recommend to use a Python environment manager such as Anaconda or similar, in order to avoid cluttering the OS's python installation. We assume you are a member of the [Github sdss organization](https://github.com/sdss). We also assume that you have an SSH key configure on your local machine and registered in your Github account. If not, please follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to set up one.

If you are planning on installing the DRP on a different OS, please read the [troubleshooting section](#troubleshooting) before you continue with the steps below.

To install the DRP along with its dependencies, you need to run the following steps:

1. Create a **Python 3.8** environment. This is optional, but strongly recommended. With conda this is done like this:
    > `conda create -n lvmdrp python=3.8`

2. Make sure you are in the intended **Python 3.8** environment and directory:
    > `conda activate lvmdrp`

3. Clone the Github repository:
    > `git clone --recurse-submodules -j8 git@github.com:sdss/lvmdrp.git` # assuming you have an SSH key

    > `git clone --recurse-submodules -j8 https://github.com/sdss/lvmdrp.git` # using HTTPS

4. Go into the lvmdrp directory:
    > `cd lvmdrp`

5. Switch to the (current) development branch:
    > `git checkout development`

6. Install the DRP package in the current python environment (see [contributing](#contributing-to-lvm-drp-development) section below for a replacement of this step):
    > `pip install --no-cache-dir . 2>&1 | tee today.txt`

## Testing the installation

<!-- write a script to test everything finished correctly with the installation -->

## Advanced ESO sky model configuration

If you are planning on using the sky module, you will need to install the ESO routines first. In order to install to do so
you need to run the following commands, **also within DRP python environment**.

1. Install the ESO skycorr and skymodel routines:
    > `drp sky installESOSky`

2. Run the sky module configuration:
    > `drp sky configureSkyModel`

## Creating Test Data

We encourage the reader to use the [LVM data simulator](https://github.com/sdss/lvmdatasimulator) to generate data for testing the DRP. But if you want to skip that step, we have already some simulations produced using the same simulator, so you don't have to run the simulator yourself, which can be computationally demanding in the case of 2D simulations.

If you follow the [examples](#examples) below, you will have access to the above mentioned simulations.

## Examples

You will find tutorial notebooks to run different DRP routines in the [examples](https://github.com/sdss/lvmdrp/tree/development/examples) folder. Here is you will find Jupyter Notebooks that illustrate different tasks in the DRP:

- [Basic Calibration](): reduction of calibration images: bias, dark and pixel flats; as well as reduction of arcs and fiber flats.
- [Wavelength Calibration](): automatic pixel to wavelength mapping and wavelength and LSF fitting.
- [Flux Calibration](): conversion of the extracted spectra from electrons to flux calibrated spectra.
- [Sky Module](https://github.com/sdss/lvmdrp/tree/development/examples/sky_module): several procedures to sky-subtract science spectra.

In each of the links above you will find a short description of what's going on in each example and also the order in which those are intended to be followed.

## Contributing to LVM-DRP development

There are two ways in which you can contribute:

- Testing the DRP and reporting bugs on Github or
- By diving into the code to fix bugs and implement new features

For those willing to contribute by coding, there are some steps to streamline the development process:

1. Before you start coding on a new feature/bug-fix, make sure your **local** `development` branch is up to date:
    > `git pull development`

2. Create a branch to work on and make sure the name can be easily mappable to the work you intend to do:
    > `git checkout -b <feature_name>`

3. Afterwards, you can push your updates to the remote branch on Github:
    > `git push`

4. Finally, if you consider your feature is ready to be merged to the `development` branch, you can create a new [pull request at Github](https://github.com/sdss/lvmdrp/pulls).

Regarding commits, I'm trying to go for an *atomic* approach, where each commit has a single purpose. So please try to avoid as much as possible pushing lots of unrelated changes in one commit.

## Troubleshooting

In some MacOS versions there may be the need to perform extra installation steps, before getting into the steps described in the [installation section](#installation).

### For MacOS (Monterey v12.6.2)

You will require to run this extra step before continuing with the regular DRP installation:
> `sudo port install py38-healpy`

See [healpy documentation](https://healpy.readthedocs.io/en/latest/install.html#compilation-issues-with-mac-os) for a statement on this issue.

After this step, you should be able to proceed with the DRP installation as described in the [installation section](#installation).

### For MacOS (Mojave v10.14.6)

The installation of the `scipy` package (a core dependency of the DRP) requires openBLAS to be installed to be able to compile the source files. If you are running on an old MacOS version, please follow these steps:

1. Install `openBLAS` by doing:
    > `brew install openblas`

2. Set `$PKG_CONFIG_PATH` to point to your installation of `openBLAS`. This may look like this:
    > `export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"`

After these steps, you should be able to proceed with the DRP installation as described in the [installation section](#installation).
