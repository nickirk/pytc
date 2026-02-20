---
description: How to add a new Jastrow factor to pytc
---
# Adding a New Jastrow Factor

This workflow guides an agent through the process of implementing a new Jastrow factor in the `pytc` repository. Jastrow factors in `pytc` are typically parameterized scalar functions of electron coordinates that multiply the reference wavefunction to introduce electron correlation.

## 1. Create the Class
New Jastrow factors should be added to the `pytc/jastrow/` directory.

1.  **Inherit from `Jastrow`**: Your new class must inherit from `pytc.jastrow.base.Jastrow` (or a similar base class depending on the specific type of Jastrow).
2.  **Required Methods**:
    *   `init_params(self) -> dict`: Returns a dictionary of initial parameters (typically JAX arrays).
    *   `evaluate(self, params: dict, coords: jax.Array) -> jax.Array`: The core evaluation function. It takes parameters and electron coordinates and returns the log of the Jastrow factor. Ensure functions that you use are `jax.jit`-compatible.
3.  **Type Hints & Shapes**: Explicitly document the expected shape of `coords` (e.g., `(n_elec, 3)`) and the returned array (e.g., scalar `()`) in the docstring of `evaluate`.

## 2. Register/Import the Class
Expose your new Jastrow factor in `pytc/jastrow/__init__.py` so it can be easily imported by users.

## 3. Implement Forward Laplacians (Optional but Recommended)
If your Jastrow factor is complex, rely on `folx` for the forward Laplacian calculation. If you need a custom implementation for performance, override the relevant `folx` rules or provide a custom derivative function, ensuring it maps correctly to the expected interface.

## 4. Add Unit Tests
Create a dedicated test file in `pytc/jastrow/test/` (e.g., `test_my_jastrow.py`).

1.  **Test Initialization**: Ensure `init_params` returns the correct keys and shapes.
2.  **Test Evaluation**: Test `evaluate` with dummy coordinates and parameters. Compare against a known analytical result or a simpler NumPy implementation if available.
3.  **Test Gradients/Laplacians (Crucial)**: Use `jax.test_util.check_grads` or a finite difference numerical check to verify that the autodiff gradients and Laplacians (via `folx`) are correct. This is the most critical step for any new physical ansatz.

## 5. Verify
Run the test suite:
```bash
python -m unittest discover pytc/jastrow/test
```
Wait for the tests to pass before proceeding to create a PR.
