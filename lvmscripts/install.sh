pip3 install numpy scipy astropy pyyaml requests fitsio matplotlib
h5py speclite

cd desiutil; python3 setup.py install
cd desimodel; python3 setup.py install

cd desici; python3 setup.py install
cd desicmx; python3 setup.py install
cd desietc; python3 setup.py install
cd desihub.github.io; python3 setup.py install

cd desidatamodel; python3 setup.py install
cd desimodules; python3 setup.py install
cd desisim-testdata; python3 setup.py install
cd desisim; python3 setup.py install
cd desispec; python3 setup.py install
cd desisurvey; python3 setup.py install
cd desitarget; python3 setup.py install
cd desitemplate; python3 setup.py install
cd desitest; python3 setup.py install

cd fiberassign; python3 setup.py install
cd nightwatch; python3 setup.py install
cd prospect; python3 setup.py install
cd redrock-templates; python3 setup.py install
cd redrock; python3 setup.py install
cd simqso; python3 setup.py install
cd specex; python3 setup.py install
cd specsim; python3 setup.py install
cd specter; python3 setup.py install
cd src; python3 setup.py install
cd surveysim; python3 setup.py install
cd teststand; python3 setup.py install
cd tilepicker; python3 setup.py install
cd tutorials; python3 setup.py install

mkdir $DESIOUT
