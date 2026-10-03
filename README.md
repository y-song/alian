# alian
Specific heppyy use for ALICE analysis

# quick start

```
git clone git@github.com:matplo/yasp.git
cd yasp
./yaspenv.sh
yasp -mi bundle/hepbase

# assuming all good
module load bundle/hepbase

git clone git@github.com:matplo/heppyy.git
cd heppyy
./install_with_yasp.sh

module load heppyy

git clone git@github.com:matplo/alian.git
cd alian
./install_with_yasp.sh

module load alian
```

# recompile
```
./install_with_yasp.sh
module load alian
```

