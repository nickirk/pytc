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
- [ ] VMC, based on Jax, for Jastrow optimization
      - [x] det.py is completed. Need tests.
      - [ ] Jax autodiff for new jastrows, sampling and optimization.

- [ ] Further efficiency improvements: 
    - [x] Identify the most time-consuming parts of the code. ~~_calc_delta_U~~
    - [ ] all eisums need careful inspection to see if they can be optimized by matrix multiplication.
        - Some of most time consuming ones are checked and optimized. But extensive tests are needed to identify more.
    - [ ] K1 and K2 mats are related by a simple indices transpose. No need to calculate K2. Just use K1 to construct
          K1+K2
    - [ ] ISDF needs similar efficiency checks, aiming at scaling to large molecules and solids, when combined with efficient 2nd quantized methods, should be competitive to real space VMC and DMC.

- [ ] GPU acceleration
