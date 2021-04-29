lvmdev () {
    export LVMHUB=$HOME/sdss5/lvmhub
    export DESIOUT=$HOME/sdss5/desiout
    export LVMMODEL=$LVMHUB/lvmifusim/lvmData

    # This is the install location of our desi software.
    # If you are not at NERSC, then change this to something
    # without "NERSC_HOST" in the name.
    desisoft="${LVMHUB}/lvmdrp/desihub"

    # Set environment variables
    export CPATH=${desisoft}/include:${CPATH}
    export LIBRARY_PATH=${desisoft}/lib:${LIBRARY_PATH}
    export LD_LIBRARY_PATH=${desisoft}/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=$HOME/.anaconda3/bin/python

    # Special setup for redmonster
    red="${HOME}/git-${NERSC_HOST}/redmonster"
    export PYTHONPATH=${red}/python:${PYTHONPATH}
    export REDMONSTER_TEMPLATES_DIR=${red}/templates

    # Choose what data files to use- these locations
    # are for NERSC.
    export DESI_ROOT=$HOME/sdss5/lvmhub/lvmdrp/desihub/desisim-testdata/desi
    export DESIMODEL=$HOME/sdss5/desimodelDATA/0.15.0 # Updated
    export DESI_BASIS_TEMPLATES=${DESI_ROOT}/spectro/templates/basis_templates/v3.2 # Updated
    export STD_TEMPLATES=${DESI_ROOT}/spectro/templates/star_templates/v1.1/star_templates_v1.1.fits
}
