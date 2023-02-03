# Local Volume Mapper (LVM) Data Reduction Pipeline (DRP)

The LVM DRP in it's current incarnation installs a collection of routines which make use of the [Py3D]().

## Installation

This code is being developed/tested in a Ubuntu-based OS. We recommend to use a Python environment manager such as Anaconda in order to avoid cluttering the OS's python installation.

To install the DRP along with its dependencies, you need to run the following steps:

1. Make sure you are in the intended python environment and directory.
   
2. Clone the Github repository:
> `git clone --recurse-submodules -j8 https://github.com/sdss/lvmdrp.git`

3. Go into the lvmdrp directory:
> `cd lvmdrp`

4. Switch to the development branch:
> `git checkout -b development`

5. Install the DRP package in the current python environment:
> `pip install .`

6. Run the ESO sky model configuration:
> `drp sky configureSkyModel`

The installation (step 5) may take a while, since it is going to install the necessary routines to run the ESO skycorr and the ASM.
The sky model configuration (step 6) should be fast as it is going to write the default configuration files in order to produce
sky models. For more advanced (and slower) sky model configuration options, see the section below.

### Advanced ESO sky model configuration

...

## Tutorials

<!-- write tutorial notebooks -->
You will find tutorial notebooks to run different DRP routines in the [examples]() folder. Here is a list of the tutorials

## Testing the installation

<!-- write a script to test everything went find with the installation -->

## Creating Test Data
