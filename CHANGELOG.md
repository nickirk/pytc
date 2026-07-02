# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
