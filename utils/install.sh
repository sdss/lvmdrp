conda activate lvmdrp
pip3 install numpy scipy astropy pyyaml requests fitsio matplotlib
pip3 h5py speclite

pip3 install -e desiutil --user
pip3 install -e specter --user
pip3 install -e desicmx --user
pip3 install -e desidatamodel --user
pip3 install -e desietc --user
pip3 install -e desimodel --user
pip3 install -e desisim --user
pip3 install -e desispec --user
pip3 install -e desisurvey --user
pip3 install -e desitarget --user
pip3 install -e desitemplate --user
pip3 install -e fiberassign --user
pip3 install -e prospect --user
pip3 install -e redrock --user
pip3 install -e surveysim --user

mkdir $DESIOUT
