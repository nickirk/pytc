# pytc examples

A numbered walkthrough of the manuscript's methodology on a single small
molecule (H2O), from Jastrow optimization through to the FNO-truncated
xTC-CCSD active-space study. Each script is runnable standalone (no ordering
dependency beyond what's noted below) and prints an expected-value
self-check.

| Script | What it shows | Manuscript section |
|---|---|---|
| `01_vmc_optimize_jastrow.py` | Two-phase reference-variance VMC optimization of a Jastrow factor | VMC / Jastrow optimization |
| `02_load_and_average_jastrow_params.py` | Polyak-Ruppert averaging of the phase-B parameter trajectory | VMC / Jastrow optimization (averaging) |
| `03_dense_xtc_ccsd.py` | Exact (dense, non-ISDF) transcorrelated CCSD | The transcorrelated (xTC) Hamiltonian and CCSD integration |
| `04_isdf_xtc_ccsd.py` | ISDF-approximated xTC-CCSD, vs. 03's dense reference | Interpolative Separable Density Fitting (ISDF) |
| `05_make_fno_xtc_ccsd.py` | FNO (MP2 natural orbital) virtual-space truncation scan, via ISDF | Frozen/truncated natural orbitals (FNO) |
| `06_rank_m_x_factor_direct_ccsd.py` | Opt-in rank-M orbital X and factor-direct ISDF xTC-CCSD | Rank-M X approximation / direct-tile CCSD |

A future example -- a deterministic, non-stochastic alternative to `01` --
is deferred. It relied on a Jastrow optimizer that was found to be broken on
`main` (a downstream host-array materialization severs its JAX gradient path)
and has been removed from `main` pending a fix; see the tracked bug (task #49).
The example will be added once the optimizer is reintroduced.

## Two honestly-separate threads

This sequence is deliberately **two short threads sharing one molecule**,
not one continuous pipeline -- see "Why 03-05 don't reuse 01/02's optimized
Jastrow" below for why they can't be merged without making the sequence
impractically slow off-cluster:

- **`01`/`02`**: how to VMC-optimize a flexible production-style Jastrow
  (`CompositeJastrow([NuclearCusp, BoysHandy])`) and average its trajectory.
- **`03`-`05`**: the dense -> ISDF -> FNO xTC-CCSD pipeline, using a plain
  `REXP` Jastrow (fixed `alpha=0.4`) instead of `01`/`02`'s optimized
  parameters.

## Why 03-05 don't reuse 01/02's optimized Jastrow

BoysHandy's dense (real-space quadrature) two-body construction is
impractically slow on a laptop CPU -- the production runs that use it were
all on GPUs -- so `03`-`05` use the fast single-parameter `REXP` correlator
instead. Measured wall times:

| System | Jastrow | Path | Grid points | Wall time |
|---|---|---|---|---|
| H2O/cc-pVDZ | REXP | dense | 21,952 | 150s |
| H2O/cc-pVDZ | REXP | ISDF (rank=15n_orb) | 21,952 | 48s |
| H2O/cc-pVDZ | BoysHandy+NuclearCusp | dense | 21,952 | killed after 35+ min |
| He/cc-pVDZ | BoysHandy+NuclearCusp | dense | 4,488 | killed after 4m44s (>19x REXP) |
| H2/cc-pVDZ | BoysHandy+NuclearCusp | dense, grid_lvl=2 | -- | killed after ~20 min |
| H2/cc-pVDZ | BoysHandy+NuclearCusp | dense, grid_lvl=1 | -- | killed after ~9.5 min |

## Ordering

- `01` writes `h2o_phase_b_hist.h5` (not committed to git -- see
  `.gitignore`). `02` reads it and prints the Polyak-Ruppert-averaged
  parameters (not consumed elsewhere -- see the split above).
- `03`-`06` are each fully standalone (hardcoded `JASTROW_PARAMS`, no shared
  state, no ordering dependency).

## Production script analogues

These examples mirror the same pipeline used for the manuscript's actual
results, at H2O/small-basis scale instead of H-chain/cc-pV5Z or
benzene/cc-pCV5Z scale:

| Example | Production analogue (tc-isdf-data repo) |
|---|---|
| `01` | `hchain/scripts/run_opt.py` |
| `02`'s averaging | `load_averaged_jastrow_params()` in `hchain/scripts/isdf_xtc_fno.py` |
| `05`'s FNO scan | `hchain/scripts/isdf_xtc_fno.py` / `isdf_xtc_fno_damped.py` |
