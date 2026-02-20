# Agent Guidelines

Welcome to the `pytc` repository! This document contains rules and guidelines for AI agents and coding assistants working in this codebase.

## Core Technologies
- **JAX**: This project relies heavily on JAX for array operations, autodiff, and GPU acceleration.
  - Prioritize `jax.numpy` (`jnp`) over standard `numpy` (`np`).
  - Be mindful of JAX's functional programming constraints (pure functions, no side effects).
- **Flax**: Used for neural network components.
- **Folx**: Used for computing forward Laplacians.
- **PySCF**: Used for handling molecular integrals and mean-field reference states.

## Code Style & Conventions
1. **Type Hints**: Use strict type hints for all function arguments and return values. For JAX arrays, use `jax.Array` or specific shapes in docstrings when helpful.
2. **Docstrings**: Use Google-style docstrings. Document the shapes of tensor inputs and outputs, as tensor contractions are complex and shape mismatches are a common source of bugs.
3. **Immutability**: Treat all JAX arrays as immutable. Use `jnp.where`, `jax.ops.index_update` (or `.at[].set()`), etc., instead of in-place modifications.

## Understanding the Architecture
Please refer to `ARCHITECTURE.md` for a textual description of how the `pytc` modules (ansatz, jastrow, vmc, xtc, df, kmat) interact.

## Performance and OOM Considerations
When adding new features, parameters, or heavy-lifting tensor operations (especially in `xtc.py`, `df.py`, or `kmat.py`), you **MUST** consider GPU memory limits. VRAM is easily exhausted by $O(N^4)$ arrays.
- **Batching & Scanning**: Utilize `jax.lax.scan` or `jax.vmap` with chunking to prevent large intermediate arrays from materializing all at once.
- **Memory Profiling**: Ensure algorithms scale gently with basis set size.
- **Example Files**: For any new feature that heavily impacts hardware performance or adds significant functionality, you must add an executable script in `pytc/examples/` demonstrating its usage cleanly (e.g., `new_feature_example.py`). Include comments explaining how to run it so users can reproduce your benchmarks without OOMs.

## Workflows
See the `.agents/workflows/` directory for specific step-by-step procedures for common tasks:
- Adding a new Jastrow factor.
- Creating a Pull Request.

## Testing
- Tests are located in `pytc/test/`, `pytc/ansatz/test/`, `pytc/vmc/test/`, `pytc/jastrow/test/`, and `pytc/solver/test/`.
- Ensure you run the relevant submodule tests before proposing code changes.
