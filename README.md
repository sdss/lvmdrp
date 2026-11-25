# Local Volume Mapper (LVM) Data Reduction Pipeline (DRP)

The LVM DRP reduces the raw data of the Local Volume Mapper into calibrated, sky-subtracted spectra. It also makes the master calibrations it needs. It is based on a collection of routines from [Py3D](https://github.com/brandherd/Py3D).

This page covers installing the DRP, reducing science exposures, running batches at Utah, and making calibrations. Longer descriptions are in [`docs/sphinx/source`](docs/sphinx/source), which GitHub displays directly:

- [Pipeline overview](docs/sphinx/source/Overview.rst): every step of a science reduction and the products it writes
- [Directory structure](docs/sphinx/source/DirectoryStructure.rst): where the inputs, calibrations and products are
- [Calibrations](docs/sphinx/source/Calibration.rst): how the master calibrations are made and selected
- [Summary file](docs/sphinx/source/SummaryFile.rst): the drpall file, column by column
- [Installation](docs/sphinx/source/Installation.rst): installation, in more detail

## Prerequisites

This code is being developed/tested in an Ubuntu-based OS, using **Python 3.10**. We recommend you use a Python environment manager such as Anaconda or similar, in order to work on the same python version and to avoid cluttering the OS's python installation. We assume you are a member of the [Github sdss organization](https://github.com/sdss). We also assume that you have an SSH key set up on your local machine and registered in your Github account. If not, please follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to set up one.

To properly install and run the DRP you'll need to follow these steps first:

1. Download the current version of [LVM Core](https://github.com/sdss/lvmcore):

    ```bash
    git clone git@github.com:sdss/lvmcore.git
    ```

    and set the environment variable `LVMCORE_DIR` pointing to the root directory `lvmcore` in your `.bashrc` (or equivalent):

    ```bash
    export LVMCORE_DIR="path/to/lvmcore"
    ```

2. Define this environment variable in your `.bashrc` (or equivalent) to point to your local mirror of the SAS:

    ```bash
    export SAS_BASE_DIR="path/to/sas-root-directory"
    ```

    you can download a target <mjd> from the SAS while preserving the directory structure using this command:

    ```bash
    wget -X css --reject html -nH -nc -t0 -r –level=2 -E –ignore-length -x -k -p -erobots=off -np -N https://data.sdss5.org/sas/sdsswork/data/lvm/lco/<mjd>/ --user <user> --password <password>
    ```
    **NOTE: we strongly recommend that you use the [SDSS access](https://github.com/sdss/sdss_access) product to achieve the same results.**

3. Create a new python environment. This is optional, but strongly recommended. With conda this is done like this:

    ```bash
    conda create -n lvmdrp python=3.10
    ```

4. Make sure you are in the intended python environment and directory:

    ```bash
    conda activate lvmdrp
    ```

## Installation

If you are planning on installing the DRP on a system other than Ubuntu (e.g., MacOS), please read the [troubleshooting section](#troubleshooting) before you continue with the steps below.

To install the DRP along with its dependencies, you need to run the following steps:

1. Clone the Github repository:

    ```bash
    git clone git@github.com:sdss/lvmdrp.git
    ```

2. Go into the `lvmdrp` directory:

    ```bash
    cd lvmdrp
    ```

3. Install the DRP package in the current python environment (see [contributing](#contributing-to-lvm-drp-development) section below for a replacement of this step). Make sure you're back in the lvmdrp directory.

    ```bash
    cd lvmdrp
    pip install .
    ```

## Testing the installation

There is a tool to quickly verify that all the needed environment variables are in place. You can run it like this:

```bash
envcheck
```

if the variables are correctly set, you should see the values of each and a successful message.

## Setup Calibration Files

Download the current set of calibrations from the [SAS sandbox](https://data.sdss5.org/sas/sdsswork/lvm/sandbox/calib/).  After installation of the pipeline, you can use the command `drp get-calibs`.  For usage, run `drp get-calibs --help`. For
example, to download all the calibration files for 60255, run

```bash
drp get-calibs -m 60255
```

This command will download the files using `sdss-access` and place them in `$LVM_MASTER_DIR`, which is defined by the
pipeline as `$LVM_SANDBOX/calib`, mirroring the SAS.  These are defined automatically relative to your root `$SAS_BASE_DIR`.
You would find the files at `$SAS_BASE_DIR/sdsswork/lvm/sandbox/calib/`.

You can download all calibration epochs by running

```bash
drp get-calibs
```

**NOTE:** If you want to know what calibration epoch is needed for a specific exposure, always choose the epoch that is closest and earlier than your exposure's MJD.


## Running the DRP

To reduce all the science exposures of a night `<mjd>`, run:

```bash
drp run -m <mjd>
```

or you can reduce a single exposure number by running:

```bash
drp run -e <expnum>
```

or several exposures: a list (`-el <expnum> -el <expnum>`), a range (`-er <first>-<last>`) or a file with one exposure number per line (`-F <expnum_file>`):

```bash
drp run -F <expnum_file>
```

You can skip steps that have already been done, for example:

```bash
drp run -e <expnum> -2d -1d -wc -fl -sk -da
```

skips the 2D reduction, the 1D reduction, the wavelength calibration and fiber flat-fielding, flux calibration, the sky subtraction and the drpall summary. The flags are listed with the reduction steps below.

Other useful options:

- `-f <method>`: the flux calibration: `MOD` (default; standard stars matched to model spectra), `STD` (standard stars), `SCI` (field stars in the science IFU) or `NONE`
- `-c`, `--clean-ancillary`: remove the intermediate products (the `ancillary` folder) once the exposure is done
- `--force-run`: also reduce exposures flagged BAD by the raw-data QA, which are otherwise skipped
- `--force-excluded`: also reduce exposures listed in `$LVMCORE_DIR/etc/exclude-exposures.yaml`, which are otherwise skipped

More options are available, you can see them by running:

```bash
drp run --help
```

Running the DRP requires that you have correctly setup your environment by following the instructions in the [Prerequisites](#prerequisites) and [Installation](#installation) sections.

The `drp run` will reduce your target exposure. Here is a list of reduction steps carried out by the DRP:

- 2D reduction steps (`-2d` flag):
  - **Preprocessing**: overscan trimming and subtraction and pixel masking
  - **Detrending**: bias and dark subtraction, Poisson error calculation, flatfielding (pixel level, when available), units conversion (e-/s)
  - **Astrometry**: adds astrometry to the primary header and RA and DEC for each fiber to the slitmap extension
  - **Stray light**: modelling and subtraction of the straylight field
- 1D reduction steps (`-1d` flag):
  - **Extraction**: fiber fitting spectra extraction, takes into account thermal fiber shifts in the Y direction
- Wavelength calibration and fiber flat-fielding steps (`-wc` flag):
  - **Spectrograph combination**: row-stacking of spectrograph fibers
  - **Wavelength calibration**: pixel-to-wavelength mapping and LSF function per fiber
  - **Fiberflat**: flatfielding (fiber level) using the master twilight fiberflat of the calibration epoch, including its spectrograph factors (see [Making calibrations](#making-calibrations))
  - **Wavelength refinement**: refines the wavelength solution by matching the sky line positions, takes into account fiber thermal shifts in the wavelength direction, only used to subtract sky from standard fibers
- Wavelength resampling steps (`-wr` flag):
  - **Sky fibers interpolation**: sky fibers interpolation along fiber ID by fitting the supersampled sky spectrum, per sky telescope; the two sky telescopes are then combined with configurable weights
  - **Wavelength resampling**: wavelength resampling to a common grid (0.5 Angstrom)
- Flux calibration steps (`-fl` flag):
  - **Flux calibration**: sensitivity curves from the standard stars and from field stars in the science IFU (reference spectra from Gaia XP, cached in `$LVM_MASTER_DIR/gaia_cache`), with a telluric correction; the method chosen with `-f` is applied to the science fibers
  - **Channel combination**: stitching together spectrographs' channels
- Sky subtraction steps (`-sk` flag):
  - **Sky subtraction**: final sky subtraction separating sky lines and continuum and combining into master sky in a predefined way
- `drpall` summary generation steps (`-da` flag):
  - **Generate/update summary**: adds a new record to the summary file (see description below)

The main outputs will be stored in the SAS directory:

```bash
$SAS_BASE_DIR/sdsswork/lvm/spectro/redux/<drpver>/<tilegrp>/<tileid>/<mjd>/
```

where `<drpver>` is `$LVMDRP_VERSION` if it is set, otherwise the version of the installed package (see step 8 of [Contributing](#contributing-to-lvm-drp-development)). The calibrations are read from `$LVM_MASTER_DIR`.

where you should find your `lvmCFrame-<expnum:08d>.fits` file, the `raw_metadata.hdf5` file and the `ancillary` folder. Within `ancillary` you'll find files following the naming conventions:

- `lvm-[pdxlwhs]object-<camera>-<expnum:08d>.fits`
- `lvm-dstray-<camera>-<expnum:08d>.fits`: contains the stray light modelling information
- `lvm-[wh]sky_[ew]-<camera>-<expnum:08d>.fits`: contains the supersampled sky fitting

where each letter in **`pdxlwh`** stands for preprocessed, detrended, extracted, wavelength-calibrated, wavelength-resampled, respectively. **`ew`** refers to east and west sky telescopes, respectively.

The main products of the pipeline:

- `lvmFrame-<channel>-<expnum:08d>.fits`: extracted, spectrograph stacked, wavelength calibrated and flatfielded frame in electrons
- `lvmFFrame-<channel>-<expnum:08d>.fits`: flux calibrated frame in physical units
- `lvmCFrame-<expnum:08d>.fits`: channel combined flux calibrated frame
- `lvmSFrame-<expnum:08>.fits`: sky subtracted frame
- `drpall-<drpver>.h5`: summary file of the reductions using `<drpver>` version of the DRP

To check a reduction step by step, `drp qa-science -e <expnum>` writes an HTML dashboard (detrending, stray light, fiber and sky-line shifts, the IFU, flux calibration, sky subtraction).

## Batch runs at Utah

At Utah, `drp cluster` submits a Slurm job that runs `drp run` for each night or exposure given:

```bash
drp cluster -F <expnum_file> --drp-options "<options for drp run>"
drp cluster -r <first_mjd>-<last_mjd> -n <nodes> -p <cores_per_node> -w <walltime>
```

Each job runs with `--skip-drpall`, so that the jobs don't write the summary file at the same time. Once the jobs are done, make the summary file in one go:

```bash
drp summary --drp-version <drpver>
```

(`--output-dir` writes it elsewhere, e.g. without write access to the SAS redux directory.)

## Making calibrations

Most users download the calibrations (see [Setup Calibration Files](#setup-calibration-files)). This section is for making them; [Calibration.rst](docs/sphinx/source/Calibration.rst) describes each master and how it is made.

The master calibrations are made per **calibration epoch**: a range of nights that share one set of masters. The epochs, and the nights whose exposures each master uses, are defined in `$LVMCORE_DIR/calibrations/calibration-epochs.yaml` (check it with `drp calibrations validate-epochs`).

1. **The masters of an epoch** (bias, fiber traces, wavelength and LSF, twilight fiberflats):

    ```bash
    drp calibrations long-term -m <epoch> --from-epochs --dry-run    # lists the exposures each master will use
    drp calibrations long-term -m <epoch> --from-epochs
    ```

    The masters are written to `$SAS_BASE_DIR/sdsswork/lvm/spectro/redux/<drpver>/0011XX/11111/<epoch>/calib/`, with QA plots in its `qa/` folder. `-b`, `-t`, `-w`, `-f` skip the bias, traces, wavelengths or twilight flats; each step uses the masters before it, so redoing one means redoing the ones after it. `-s` skips steps already done. A full epoch takes about two days on one Utah node.

2. **The spectrograph factors of the twilight fiberflats.** The factors between the three spectrographs, and an IFU gradient, are measured from reduced science exposures of each fiber-flat factor epoch (`$LVMCORE_DIR/calibrations/fiberflat-factor-epochs.yaml`) and applied to the twilight fiberflats:

    ```bash
    drp calibrations fiberflat-corrections -m <epoch>
    ```

    `--undo-correction` removes them again.

3. **Tagging.** `drp calibrations create-tag -m <epoch> --version <version>` copies the masters into `$LVM_MASTER_DIR/<epoch>/` (normally `sandbox/calib`, which the DRP reads), **replacing the files there**, and then snapshots that epoch into `sandbox/calib_<version>/<epoch>/`.

Several epochs can be run as one Slurm job: `drp calibrations cluster --routine long-term --from-epochs -r <first>-<last>`.

The calibrations have their own QA commands: `drp calibrations qa-wavelengths`, `qa-arc-sky`, `qa-twilights`, `qa-domes`, `qa-fiberflats`, `qa-pixelflats`, `qa-raw-pixelflats`; `drp calibrations <command> --help` describes each.

## ESO sky routines installation and configuration

**NOTE: you don't need to install the ESO sky routines to be able to run the science reductions as shown in the previous section**

If you are planning on using the sky module, you will need to install the ESO routines first. To install [skycorr](https://www.eso.org/sci/software/pipelines/skytools/skycorr) and the [ESO Sky Model](https://www.eso.org/sci/software/pipelines/skytools/skymodel), follow the instructions in the following links:

- https://wiki.sdss.org/display/LVM/ESO+skycorr
- https://wiki.sdss.org/display/LVM/ESO+Sky+Model

respectively. Additionaly, you'll need to set following the environment variable in your `.bashrc` (or equivalent):

```bash
export LVM_ESOSKY_DIR="path/to/eso-routines"
```

where `eso-routines` is a directory containing the root directories of both, the *skycorr* and the *ESO sky model* installations.

## Creating test data

We encourage the reader to use the [LVM data simulator](https://github.com/sdss/lvmdatasimulator) to generate data for testing the DRP. But if you want to skip that step, we have already some simulations produced using the same simulator, so you don't have to run the simulator yourself, which can be computationally demanding in the case of 2D simulations.

If you follow the examples below, you will have access to the above mentioned simulations.

## Examples

The [examples](examples) folder has Jupyter notebooks that illustrate tasks in the DRP. Most were written for earlier versions (the [calibration](examples/calibration) notebooks in 2023, the [sky module](examples/sky_module) in 2024) and may not run with the current code.

## Contributing to LVM-DRP development

There are two ways in which you can contribute:

- Testing the DRP and reporting bugs on Github or
- By diving into the code to fix bugs and implement new features

For those willing to contribute by coding, there are some steps to streamline the development process:

1. Make sure you install the pipeline on your environment in edit (developer) mode, like this:

    ```bash
    pip install -e .'[dev]'
    ```

2. Before you start coding on a new feature/bug-fix, make sure your **local** `dev` branch is up to date (`dev` is the development branch; `main` holds the released versions):

    ```bash
    git checkout dev
    git pull origin dev
    ```

3. Create a branch to work on and make sure the name can be easily mappable to the work you intend to do:

    ```bash
    git checkout -b <feature_name>
    ```
4. Start coding. Once you're done implementing changes:

   ```bash
   git status #check what has changed and identify the files you want to commit
   git add <changed_files>
   git commit -m "commit message"
   ```

6. Afterwards, you can push your updates to the remote branch on Github:

    ```bash
    git push
    ```

7. If you consider your feature is ready, create a [pull request](https://github.com/sdss/lvmdrp/pulls) to `dev`. Every pull request needs an approving review by someone other than its author. Describe the change in the pull request; `CHANGELOG.rst` is written when a version is released, not in pull requests. Run the tests (`pytest`) and the linter (`ruff check .`) first.

8. Finally, it might useful to have different versions of the DRP outputs in separated directories. To achieve this, you can set the environment variable `LVMDRP_VERSION` in your `.bashrc` (or equivalent) to a string representing the version of the DRP you are working on. For example:

    ```bash
    export LVMDRP_VERSION="my_tests"
    ```

    This will make the DRP to store the outputs in a directory named `my_tests`, under the usual SAS path:

    ```bash
    $SAS_BASE_DIR/sdsswork/lvm/spectro/redux/my_tests/<tilegrp>/<tileid>/<mjd>/
    ```

Regarding commits, we aim for an *atomic* approach, where each commit has a single purpose. So please try to avoid as much as possible pushing lots of unrelated changes in one commit.

## Troubleshooting

In some MacOS versions there may be the need to perform extra installation steps, before getting into the steps described in the [installation section](#installation).

### Issue importing CSafeLoader

Some Mac users have found the folloring error while importing `CSafeLoader` from the PyYaml package (~6.0):

```
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
