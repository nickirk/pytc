# pytc
**Py**thon **T**rans**C**orrelation package

## Dependencies
- numpy
- scipy
- pyscf (modified version)

## Features
- Takes in user defined Jastrow factors
- Calculates the transcorrelated 2-body integrals: K1, K2, K3
- Calculates the xTC approximated 3-body integrals
- Implements the Interpolative Separable Density Fitting (ISDF) approximation for the all the aforementioned integrals
- Seamless integration with PySCF rccsd solver, and more to come...

## Usage
See the test directory for examples of how to use the package.

## TODO
- [ ] Jastrow optimization VMC

- [ ] Further efficiency improvements: 
    - [x] Identify the most time-consuming parts of the code. ~~_calc_delta_U~~
    - [ ] all eisums need careful inspection to see if they can be optimized by matrix multiplication.

- [ ] GPU acceleration
