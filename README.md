# lvmdrp
Local Volume Mapper (LVM) Data Reduction Pipeline

The LVM DRP in it's current incarnation installs a collection of routines which make use of the DESI-DRP, specifically spectre and DESI-PREPROC.

The purpose of the current repository is 2 fold.

1. Automatically recreate the DESI environment, with dependencies
2. Host LVM routines/wrappers/patches to operate the based DESI-DRP

## Installation

This code is being developed in a Ubuntu-based OS. We recommend to use a Python environment manager such as Anaconda in order to avoid cluttering the OS's python installation.

To get the LVM DRP and DESI reduction package working on your system, you need to follow this steps:

1. Define the `LVMHUB` environment variable to point to the root directory containing the LVM DRP repositories. E.g. in bash:
> `export LVMHUB=path/to/drp/root`

2. `cd $LVMHUB`

3. Clone the repository:
> `git clone --recurse-submodules -j8 git://github.com/sdss/lvmdrp.git`

4. Install external (OS) dependencies listed in `$LVMHUB/lvmdrp/requirements_ubuntu.txt`:
> `sudo apt install -y $(awk '{print $1'} $LVMHUB/lvmdrp/requirements_ubuntu.txt)`

5. Run installation script:
> `bash $LVMHUB/lvmdrp/utils/install.sh`

## Testing the installation

<!-- write a script to test everything went find with the installation -->

## Creating Test Data
