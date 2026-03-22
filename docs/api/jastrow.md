# Jastrow Factors

`pytc.jastrow`

Modular Jastrow factor implementations for explicitly correlated wave functions.
All Jastrow factors follow a common interface defined by the base `Jastrow` class and are implemented as Flax dataclasses for JAX compatibility.

## Base Class

### `Jastrow`

Abstract base class for JAX-based Jastrow factors.
Parameters are not stored in the instance but passed directly to methods, aligning with JAX's philosophy for parameter handling.

**Key methods:**

`_compute(r1, r2, params)`
: Core computation of the Jastrow exponent *u* for a pair of electrons.

`u(r1, r2, params)`
: Compute the Jastrow exponent (calls `_compute`).

`grad_lapl(r1, r2, params)`
: Compute gradient and Laplacian of the Jastrow factor via `folx`.

`init_params()`
: Return default initial parameters.

## Concrete Implementations

### `BoysHandy`

Boys-Handy Jastrow factor with electron-electron, electron-nuclear, and electron-electron-nuclear terms.

```python
from pytc.jastrow import BoysHandy
jbh = BoysHandy.create(mol, name="bh")
```

### `NuclearCusp`

Nuclear cusp Jastrow factor ensuring the correct electron-nuclear cusp condition.

```python
from pytc.jastrow import NuclearCusp
jncusp = NuclearCusp.create(mol, name="ncusp")
```

### `REXP`

Radial exponential Jastrow factor with a simple parametric form.

```python
from pytc.jastrow import REXP
import jax.numpy as jnp
my_jastrow = REXP()
params = {'alpha': jnp.array([1.0])}
```

### `Poly`

Polynomial Jastrow factor.

```python
from pytc.jastrow import Poly
```

### `NeuralEN`, `NeuralEE`, `NeuralEEN`

Neural network-based Jastrow factors for electron-nuclear (EN), electron-electron (EE), and electron-electron-nuclear (EEN) correlations. Built with Flax neural networks.

```python
from pytc.jastrow import NeuralEN, NeuralEE, NeuralEEN
```

### `CompositeJastrow`

Combines multiple Jastrow factors into a single composite factor (sum of individual exponents).

```python
from pytc.jastrow import CompositeJastrow, NuclearCusp, BoysHandy
jncusp = NuclearCusp.create(mol, name="ncusp")
jbh = BoysHandy.create(mol, name="bh")
jastrow = CompositeJastrow.create([jncusp, jbh])
```
