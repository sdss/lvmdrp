#!/usr/bin/bash

#conda activate lvmdrp
pip3 install numpy scipy astropy pyyaml requests fitsio matplotlib
pip3 install h5py speclite


pref="${LVMHUB}/lvmdrp/desihub/"

cd ${pref}

for pkg in desiutil desimodel desitarget desisim specter desispec specex; do
    pip3 uninstall ${pkg}
    pip3 install -e ${pkg} --user
done

mkdir $DESIOUT

