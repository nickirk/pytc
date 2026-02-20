---
description: Create a Pull Request (PR) in pytc
---
# Creating a Pull Request (PR)

Before submitting a Pull Request to `pytc`, AI agents must follow this workflow to ensure code quality and stability.

## Prerequisites
1. **Understand Architecture:** Review `.agents/ARCHITECTURE.md` and `.agents/rules.md`.
2. **Implement Feature/Fix:** All tensor operations must have shape documentation.
3. **Write Tests:** New features or Jastrow factors require corresponding unit tests.

## 1. Run Tests Locally
You **MUST** run the relevant test suite before proposing changes:

- **Core tests:** `python -m unittest discover pytc/test`
- **Submodule tests:**
  - Jastrow (if adding/modifying Jastrow factors): `python -m unittest discover pytc/jastrow/test`
  - Ansatz: `python -m unittest discover pytc/ansatz/test`
  - VMC: `python -m unittest discover pytc/vmc/test`
  - Solver: `python -m unittest discover pytc/solver/test`

## 2. Verify Output and Types
- Do the tests pass? If not, fix the bugs before proceeding.
- Are type hints consistent? Have you used `jax.Array` appropriately?

## 3. Formatting and Linting
*If a linter is configured, document the command here. E.g., `flake8 pytc/` or `black pytc/`.*

## 4. Submitting
1. Provide a clear summary in your PR description.
2. Link any related issues or discussions.
3. State explicitly which tests were run and that they all passed.
