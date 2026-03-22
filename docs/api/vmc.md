# Variational Monte Carlo

`pytc.vmc`

VMC sampling, optimization, and analysis utilities.

## Optimization

### `optimize_ref_var`

`pytc.vmc.optimize_ref_var(ansatz, params, ...)`

High-level function for VMC-based Jastrow optimization with reference variance minimization.

**Key parameters:**

`ansatz`
: A `SlaterJastrow` ansatz.

`params`
: Initial Jastrow parameters (list of PyTrees).

`n_walkers`
: Number of Monte Carlo walkers.

`n_steps`
: Number of sampling steps per optimization iteration.

`n_opt_steps`
: Number of optimization iterations.

`optimizer_type`
: Optimizer to use (`'newton'` or `'adam'`).

**Returns:** Dictionary with numpy arrays for keys like `'cost'`, `'energies'`, and `'stds'`.

### `optimize`

`pytc.vmc.optimize(ansatz, params, ...)`

Lower-level optimization function with more control over the optimization loop.

## Sampling

### `sample`

`pytc.vmc.sample(ansatz, params, walker, ...)`

Run Metropolis-Hastings sampling to generate electron configurations.

### `burn_in`

`pytc.vmc.burn_in(ansatz, params, walker, ...)`

Equilibrate walkers before production sampling.

### `metropolis_hastings`

`pytc.vmc.metropolis_hastings(ansatz, params, walker, ...)`

Single step of the Metropolis-Hastings algorithm.

## Walker

### `Walker`

`pytc.vmc.Walker`

Container for Monte Carlo walker state. Stores electron positions and associated metadata. Exposes a `.shape` property returning `positions.shape` for compatibility. Walker PyTrees use axis-0 as `n_walkers`.

### `initialize_walkers`

`pytc.vmc.initialize_walkers(ansatz, n_walkers, ...)`

Initialize a set of random walkers for VMC sampling.

## Optimizer

### `NewtonOptimizer`

`pytc.vmc.optimizer.NewtonOptimizer`

Second-order optimizer supporting:
- **Stochastic Reconfiguration (SR)** / Natural Gradient for energy minimization (`curvature="fisher"`)
- **Gauss-Newton** for variance minimization (`curvature="gauss_newton"`)

Solvers: `"cg"` (Conjugate Gradient, iterative, matrix-free), `"exact"` / `"cholesky"` (exact matrix inversion).

### `create_optimizer`

`pytc.vmc.create_optimizer(optimizer_type, ...)`

Factory function to create an optimizer by name.

## Analysis

### `block_analysis`

`pytc.vmc.block_analysis(data, ...)`

Perform blocking analysis to estimate statistical errors accounting for autocorrelation.

### `analyze_optimization_history`

`pytc.vmc.analyze_optimization_history(history, ...)`

Analyze and summarize optimization run results.

## Multi-GPU Sharding

### `create_mesh`

`pytc.vmc.create_mesh(...)`

Create a JAX device mesh for multi-GPU parallelism.

### `shard_walker`

`pytc.vmc.shard_walker(walker, mesh, ...)`

Shard walker data across multiple devices.
