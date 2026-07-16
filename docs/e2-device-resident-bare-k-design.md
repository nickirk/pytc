# E2: device-resident bare-K boundary

Status: design-only proposal for review.  This document introduces no code
path, default, GPU run, or timing claim.

## Scope and invariant

The proposed lane computes *bare* periodic exchange only.  It must reproduce
the existing NumPy `pytc.pbc.coulomb.get_k` calculation for valid inputs:

\[
\rho_k = X_k D_k X_k^\dagger / N_k,\quad
\rho_R = \mathop{\mathrm{Re}}(P\rho)_R^T,
\]
\[
W_R = \sqrt{N_k}\,\mathop{\mathrm{Re}}(PW)_R,\quad
V_R=W_R\odot\rho_R,\quad V_k=P^\dagger V_R,
\]
\[
K_k=\left(X_k^T V_k X_k^*\right)^*.
\]

Here `X=inpv_kpt`, `W=coul_kpt`, and `P=phase`.  The `real` operations have
the same time-reversal validity meaning as `kpt_to_spc`; they are never a
silent approximation.  Ewald is deliberately outside this boundary.

The existing public `get_k` remains the NumPy oracle and its behavior is not
changed in this stage.  `ISDFDF.get_jk` likewise remains on that oracle until
a separately reviewed integration change.

## Current residency and transfers

`build_coul_kpt_device` returns `coul_kpt` as a JAX c128 array.  `build`
returns it together with a host NumPy `inpv_kpt`, host NumPy
`KptsMesh.phase`, and host integer `KptsMesh.neg`.  Today `get_k` immediately
does the following:

| value | source residence | current action in `get_k` | consequence |
| --- | --- | --- | --- |
| `dm_kpts` | PySCF/NumPy host | `np.asarray(..., complex128)` | host-only density projection |
| `inpv_kpt` | host NumPy | `np.asarray(..., complex128)` | stays host |
| `coul_kpt` | JAX device from S4 | `np.asarray(..., complex128)` | full device-to-host materialization |
| `phase` | host NumPy | consumed by NumPy transforms | host transform |
| `neg` | host NumPy integer | `np.asarray` and Python loop | host TR projection |
| `vk_kpts` | host NumPy | returned to PySCF | required public result form |

Thus S4's device-resident `coul_kpt` is materialized on the host solely for
the exchange consumer.  The proposal removes that transfer from the bare-K
calculation, not from the public PySCF boundary.  A future call still starts
with an unavoidable host-to-device upload of a PySCF density matrix unless
the caller already owns a device density.  The factor `X`, phase matrix, and
`neg` are uploaded at the explicit device-boundary/cache policy; the device
kernel stays resident.  A bare device API returns a JAX array.  The adapter,
when it is later authorized to use it, performs exactly one explicit
device-to-host transfer for PySCF, and only then applies host Ewald.

No cache lifetime or memory-budget policy is proposed here.  In particular,
there is no implicit global cache and no claim that repeated SCF iterations
avoid factor uploads until such a policy is separately reviewed.

## Proposed split

Add a future private JAX-only implementation rather than modifying `get_k`:

```python
def _get_k_bare_device(
    dm_kpts, inpv_kpt, coul_kpt, phase, neg=None, *, imag_tol=1e-10,
) -> jax.Array:
    """Return batched device c128 bare K; no PySCF calls or Ewald."""
```

Its public-facing wrapper is responsible for accepting only fixed-shape c128
arrays, preserving single-set versus batched shape at the boundary, and
validating host-side metadata before dispatch:

- `X.shape == (Nk, Nip, Nao)`, `W.shape == (Nk, Nip, Nip)`, and density shape
  is `(Nk, Nao, Nao)` or `(Nset, Nk, Nao, Nao)`;
- `phase.shape == (Nk, Nk)`, `neg.shape == (Nk,)` when present, and `neg` is
  an in-range involution (`neg[neg[k]] == k`);
- JAX 64-bit mode is enabled and every floating input is c128 (no implicit
  c64 downcast); and
- `exxdiv` is not an argument to the bare device API.

The wrapper may transfer only scalar diagnostics to host.  It must not call
`np.asarray` or `jax.device_get` on an array-sized device operand or result.

The jitted implementation is two device stages:

1. A preflight projects `rho` and forms `rho_spc_complex = P rho` and
   `coul_spc_complex = sqrt(Nk) P W`.  If `neg` is supplied, it constructs
   exact time-reversal representatives on device: retain index `k` for
   `k <= neg[k]`, set its partner to its conjugate, and make a self-paired
   representative real.  This is the present `get_k` rule, expressed without
   a Python loop.  Without `neg`, no projection occurs.
2. The wrapper reads only the scalar relative imaginary norms from preflight
   and enforces the existing `imag_tol` gate before continuing.  On success,
   the device core consumes the already device-resident transforms, takes
   their real parts, performs the Hadamard product, inverse phase transform,
   and final contraction above.

Keeping preflight separate preserves the NumPy oracle's "validate before
discarding imaginary components" contract rather than silently applying
`jnp.real`.  It also keeps the actual K core free of Python, NumPy, PySCF,
and host transfers.  Both stages use explicit `jnp.einsum`/matrix products
with the supplied unitary phase matrix; they do not reinterpret the canonical
k ordering as an FFT reshape.

The phase transform needs the same axis convention as today:

```python
# flatten only trailing axes
spc = (phase @ values.reshape(Nk, -1)).reshape(values.shape)
kpt = (phase.conj().T @ spc.reshape(Nk, -1)).reshape(spc.shape)
```

For each density set the core transposes the real supercell density's two
interpolation axes exactly as the oracle does before the Hadamard product.
`coul_kpt` is not symmetrized or repaired: its preflight gate is evidence
that the S4 factor already has the required time-reversal property.

## Boundary ownership

| concern | owner |
| --- | --- |
| factor construction and `coul_kpt` residency | existing S4 JAX path |
| shape/dtype/`neg` metadata validation | future host wrapper |
| TR projection and phase transforms | device preflight/core |
| imaginary-norm decision | host wrapper using device scalar diagnostics |
| bare K result | device core, c128 |
| PySCF ndarray conversion | future adapter boundary only |
| `_ewald_exxdiv_for_G0` | existing host post-processing after that conversion |

This deliberately excludes a device Ewald implementation, a public API
replacement, a fallback path, a change to the selector/S4 kernels, and a
memory/performance assertion.

## Oracle and acceptance gates before any integration

`get_k` remains the independent NumPy oracle.  A later implementation is
acceptable only after CPU-JAX c128 comparisons against it, using the same
fixed inputs and no FFTDF substitution for this parity layer:

1. Gamma-only real fixture: exact molecular reduction and both single-set and
   batched output shape conventions.
2. A genuine-pair mesh with a valid `neg` map: TR-symmetric Hermitian
   densities, Hermitian output per k point, and device/NumPy agreement at the
   existing machine tier.
3. A near-TR SCF-style density: `neg=None` preserves the imaginary gate;
   `neg` supplied projects by construction and agrees with the NumPy
   `neg` path.
4. Malformed shapes, non-involutory `neg`, disabled x64, and non-c128 input
   fail at the boundary.  Ewald is rejected by the bare API and stays covered
   by the existing host-oracle tests.

Only after those gates and review may a separate change decide whether
`ISDFDF.get_jk` selects the device path.  That change must retain the NumPy
oracle test coverage, state its explicit output transfer, and be reviewed
without treating any CPU parity result as GPU or timing evidence.
