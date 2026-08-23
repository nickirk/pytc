# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-08-21

### Added

- Deterministic blocked pivot selection for molecular ISDF.
- Capability-aware exact auxiliary recovery of the K1 and K3 kernels,
  including streamed out-of-core construction and cache provenance.
- An opt-in rank-M orbital Tucker representation for the X kernel.
- An opt-in persistent XLA compilation cache.

### Changed

- Accelerator-memory probes now fail closed when the available capacity cannot
  be measured reliably.
- Disk-backed X panels remain bounded during the JAX CCSD direct-tile path.

### Fixed

- Delta-U dispatch-time memory sizing no longer double-counts an already
  resident D kernel.
- Rank-major X-store conversion retains the source layout and adds a validated
  provenance-stamped twin for legacy compatibility.

### Validation

- On H10/R=1.6/cc-pVTZ/grid2 on one V100, using one fixed ISDF base, exact
  auxiliary recovery reduced K1/K3 construction from 324.90 s to 89.62 s
  (3.63x) and total ISDF-intermediate construction from 922.96 s to 714.25 s
  (1.29x), with K1/K3 relative errors below 1.2e-15 and normal-order
  identities agreeing within 2.4e-16.
- On H10/cc-pVTZ/grid2 on one A100 (batch size 32), blocked pivot selection
  was 9.515x faster in the relaxed-energy run; the full-X total energy changed
  by +0.000267 mHa.
- On the same H10 gate, the opt-in M=80 orbital-X approximation changed the
  exact-pivot total energy by -0.844607 mHa; combining blocked selection with
  M=80 changed it by -0.852088 mHa. Transfer beyond H10 and a matched
  rank-M construction speedup have not been established.

## [0.2.0] - 2026-08-03

### Added

- Factorized ISDF xTC-CCSD path for the JAX RCCSD solver that contracts the
  VVVV-T2 term without ever materializing the full `(nvir, nvir, nvir, nvir)`
  tensor. The large X factor is streamed one rank panel at a time.
- Three-tier residency gate for X-factor access (device lift, host-resident,
  store stream), selected from measured free memory. Cgroup-aware, fails closed
  when memory cannot be measured, and accounts for reclaimable cache.
- Optional rank-major store layout (`X_rm`) giving contiguous panel reads.
  Stores written without it behave exactly as before.
- `pytc.df` package with the panelled JAX robust DF-THC fit and sandwich
  routines (`df.py` is preserved as the package root).
- Opt-in per-term timers (`pytc.utils.tile_timers`). Disabled by default with
  zero accumulation when off.

On a 1200-orbital benzene (cc-pCV5Z/cc-pV5Z), an on-the-fly xTC-CCSD
calculation converges on a single NVIDIA B200 in about 9 hours, reproduced
bit-identically, against roughly 29 hours on 8x B200 by the previous route at
the same FP64 precision.

### Fixed

- Frozen Jastrow parameters are masked correctly during optimization.
- VMC burn-in step-size adaptation cadence.
- Latent custom-JVP `NameError` and diagonal-NaN poisoning in the loss path.
- Grid dimension is chunked in `TC.from_pyscf`, fixing an H50 out-of-memory
  failure on 2x A100.
- FNO occupancy cut is thresholded at the eigengap midpoint rather than at a
  reused occupation value.

### Changed

- Documentation moved to a standalone `pytc-docs` site; the previous URL
  redirects.
- Narrative commentary stripped from the source tree.

### Removed

- `pytc/optimize.py`. It was broken and had no users.
- `pytc/utils/perf_baseline.py`.
- Five example scripts: `Be_vmc_ref_opt_xtc_ccsd.py`,
  `benchmark_isdf_xtc_kdx.py`, `co2_simple_jastrow_xtc_ccsd.py`,
  `h2o_jastrow_xtc_isdf_ccsd.py`, `h2o_jax_isdf_xtc.py`. The numbered
  walkthrough under `examples/` replaces them.

### Upgrading

The residency gate and the per-term timers are inactive by default, guarded by
environment pins read at call time. The `X_rm` twin is optional. `psutil` is
optional, with a cgroup fallback. The only import that can break is
`pytc.optimize`, which is removed outright.

## [0.1.0] - 2026-07-01

### Added

- xTC-CCSD transcorrelated coupled-cluster with singles and doubles, including
  ISDF factorization of the two-electron integrals for reduced scaling.
- VMC-based Jastrow factor optimization via JAX autodiff.
- GPU memory auto-sizing: tile and panel sizes are determined automatically
  from the available device memory, eliminating the need to hand-set
  `PYTC_PANEL_BLK` / `PYTC_SOLVER_BLK` environment variables.
- User documentation for GPU memory management and environment-variable
  override knobs (`docs/gpu-memory.md`).
- Initial PyPI release as `pytc-qc` (import name remains `pytc`).
