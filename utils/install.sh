#!/usr/bin/bash

#conda activate lvmdrp
pip3 install numpy scipy astropy pyyaml requests fitsio matplotlib
pip3 install h5py speclite


pref=$LVMHUB/lvmdrp/desihub/

cd $pref

for pkg in desiutil desimodel specter desispec desisim desitarget; do
    pip3 uninstall --yes ${pkg}
    # we need to do this in order to install DESI dependencies since those are not included in their
    # setup.py. This is a temporary patch that will not work in the long run
    yes | pip3 install -r $pkg/requirements.txt
    yes | pip3 install -e ${pkg} --user
done

#mkdir $DESIOUT
