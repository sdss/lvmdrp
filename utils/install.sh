#!/usr/bin/bash

#conda activate lvmdrp
pip3 install numpy scipy astropy pyyaml requests fitsio matplotlib
pip3 install h5py speclite

packages=("desiutil" "desimodel" "specter" "desispec" "desisim-testdata" "desitarget")
versions=("3.2.2" "0.15.0" "0.10.0" "0.43.0" "0.6.1" "1.2.0")

pref=$LVMHUB/lvmdrp/desihub/

cd $pref

for i in "${!packages[@]}"; do
    if [ ! -d "${packages[i]}" ]
    then
        git clone --depth 1 --branch ${versions[i]} "https://github.com/desihub/${packages[i]}"
    else
        echo "${packages[i]} already exists, skipping..."
    fi
done

for pkg in ${packages[*]}; do
    pip3 uninstall --yes ${pkg}
    # we need to do this in order to install DESI dependencies since those are not included in their
    # setup.py. This is a temporary patch that will not work in the long run
    yes | pip3 install -r $pkg/requirements.txt
    yes | pip3 install -e ${pkg} --user
done

#mkdir $DESIOUT
