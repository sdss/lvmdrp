# Reading argument values using loop
clean=false
for argval in "$@"
do
  if [ "$argval" = "--clean" ]; then
    clean=true
  fi
done

if $clean; then
    echo "cleaning"
    rm -rf desici/
    rm -rf desicmx/
    rm -rf desietc/
    rm -rf desihub.github.io/
    rm -rf desimodel/
    rm -rf desimodelDATA/
    rm -rf desimodules/
    rm -rf desisim-testdata/
    rm -rf desisim/
    rm -rf desispec/
    rm -rf desisurvey/
    rm -rf desitarget/
    rm -rf desitemplate/
    rm -rf desitest/
    rm -rf desiutil/
    rm -rf fiberassign/
    rm -rf lvm-notebooks/
    rm -rf nightwatch/
    rm -rf prospect/
    rm -rf redrock-templates/
    rm -rf redrock/
    rm -rf simqso/
    rm -rf specex/
    rm -rf specsim/
    rm -rf specter/
    rm -rf src/
    rm -rf surveysim/
    rm -rf teststand/
    rm -rf tilepicker/
    rm -rf tutorials/
fi

reclone=false
for argval in "$@"
do
  if [ "$argval" = "--reclone" ]; then
    reclone=true
  fi
done

if $reclone; then

  git clone https://github.com/desihub/desici&
  git clone https://github.com/desihub/desicmx&
  git clone https://github.com/desihub/desietc&
  git clone https://github.com/desihub/desihub.github.io&
  git clone https://github.com/desihub/desimodel&
  git clone https://github.com/desihub/desidatamodel&
  git clone https://github.com/desihub/desimodules&
  git clone https://github.com/desihub/desisim-testdata&
  git clone https://github.com/desihub/desisim&
  git clone https://github.com/desihub/desispec&
  git clone https://github.com/desihub/desisurvey&
  git clone https://github.com/desihub/desitarget&
  git clone https://github.com/desihub/desitemplate&
  git clone https://github.com/desihub/desitest&
  git clone https://github.com/desihub/desiutil&
  git clone https://github.com/desihub/fiberassign&
  git clone https://github.com/desihub/lvm-notebooks&
  git clone https://github.com/desihub/nightwatch&
  git clone https://github.com/desihub/prospect&
  git clone https://github.com/desihub/redrock-templates&
  git clone https://github.com/desihub/redrock&
  git clone https://github.com/desihub/simqso&
  git clone https://github.com/desihub/specex&
  git clone https://github.com/desihub/specsim&
  git clone https://github.com/desihub/specter&
  git clone https://github.com/desihub/src&
  git clone https://github.com/desihub/surveysim&
  git clone https://github.com/desihub/teststand&
  git clone https://github.com/desihub/tilepicker&
  git clone https://github.com/desihub/tutorials&
  git clone https://github.com/sdss/lvmdrp-testdata
fi

update=false
for argval in "$@"
do
  if [ "$argval" = "--update" ]; then
    update=true
  fi
done

if $update; then
    echo "desici"; cd desici; git pull; cd ..; sleep 1;
    echo "desicmx"; cd desicmx; git pull; cd ..; sleep 1;
    echo "desietc"; cd desietc; git pull; cd ..; sleep 1;
    echo "desimodel"; cd desimodel; git pull; cd ..; sleep 1;
    echo "desimodelDATA"; #cd desimodelDATA; git pull; cd ..; sleep 1;
    echo "desimodules"; cd desimodules; git pull; cd ..; sleep 1;
    echo "desisim-testdata"; cd desisim-testdata; git pull; cd ..; sleep 1;
    echo "desisim"; cd desisim; git pull; cd ..; sleep 1;
    echo "desispec"; cd desispec; git pull; cd ..; sleep 1;
    echo "desisurvey"; cd desisurvey; git pull; cd ..; sleep 1;
    echo "desitarget"; cd desitarget; git pull; cd ..; sleep 1;
    echo "desitemplate"; cd desitemplate; git pull; cd ..; sleep 1;
    echo "desitest"; cd desitest; git pull; cd ..; sleep 1;
    echo "desiutil"; cd desiutil; git pull; cd ..; sleep 1;
    echo "fiberassign"; cd fiberassign; git pull; cd ..; sleep 1;
    #echo "lvm-notebooks"; cd lvm-notebooks; git pull; cd ..; sleep 1;
    echo "nightwatch"; cd nightwatch; git pull; cd ..; sleep 1;
    echo "prospect"; cd prospect; git pull; cd ..; sleep 1;
    #cd redrock-templates; git pull; cd ..; sleep 1;
    #cd redrock; git pull; cd ..; sleep 1;
    #echo "simqso"; cd simqso; git pull; cd ..; sleep 1;
    echo "specex"; cd specex; git pull; cd ..; sleep 1;
    echo "specsim"; cd specsim; git pull; cd ..; sleep 1;
    echo "specter"; cd specter; git pull; cd ..; sleep 1;
    echo "surveysim"; cd surveysim; git pull; cd ..; sleep 1;
    echo "teststand"; cd teststand; git pull; cd ..; sleep 1;
    echo "tilepicker"; cd tilepicker; git pull; cd ..; sleep 1;
    echo "tutorials"; cd tutorials; git pull; cd ..; sleep 1;
    echo "lvmdrp-testdata"; cd lvmdrp-testdata; git pull; cd ..; sleep 1;
fi
