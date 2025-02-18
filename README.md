# pytc
**Py**thon **T**rans**C**orrelation package

## Dependencies
- numpy
- scipy
- unittest
- jax (autodiff and GPU acceleration)
- optax (machine learning optimizers)
- tqdm (visual progress tracking)
- pyscf (![modified version](https://github.com/nickirk/pyscf/tree/tc-ccsd) for tc-ccsd only. Otherwise, official pyscf also works)

## Features
- Takes in user defined Jastrow factors
- Uses JAX autodiff to compute gradient of Jastrow on r and on parameters
- Support simple Jastrow optimization by deterministic/second quantized optimization algorithm
- Real-space VMC Jastrow optimization (coming soon)
- Supports GPU acceleration via JAX (CUDA-backend currently implemented)
- Calculates the transcorrelated 2-body integrals: K1, K2, K3
- Calculates the xTC approximated 3-body integrals
- Implements the Interpolative Separable Density Fitting (ISDF) approximation for the all the aforementioned integrals
- Seamless integration with PySCF rccsd solver, and more to come...

## Usage
See the test directory for examples of how to use the package.

## TODO
- [x] JAX autodiff is now completed and tested for simple Jastrow.
    - [x] ~~det.py is completed. Need tests.~~ Related to VMC, not relevant here. 
    - [x] Jax autodiff for new jastrows and optimization.

- [ ] Further efficiency improvements: 
    - [x] Identify the most time-consuming parts of the code. ~~_calc_delta_U~~
    - [ ] all eisums need careful inspection to see if they can be optimized by matrix multiplication.
        - Some of most time consuming ones are checked and optimized. But extensive tests are needed to identify more.
    - [x] K1 and K2 mats are related by a simple indices transpose. No need to calculate K2. Just use K1 to construct
          K1+K2
    - [ ] ISDF needs similar efficiency checks, aiming at scaling to large molecules and solids, when combined with efficient 2nd quantized methods, should be competitive to real space VMC and DMC.
    - [ ] Needs more careful memory management, by improving slicing of the grid points or even the number of orbitals.

- [x] GPU acceleration
    - [ ] In principle, now can use JAX's GPU support to accelerate integral computations (NEED MORE TESTS ON REAL GPU!)
