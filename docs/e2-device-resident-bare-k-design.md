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

Add a future private JAX-only implementation rather than modifying `get_k`.
It has a pure device boundary and no host-array conversion:

```python
@jax.jit
def _get_k_bare_device_preflight(
    dm_sets, inpv_kpt, coul_kpt, phase, neg,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """All operands/results are device arrays; no Ewald or PySCF."""

@jax.jit
def _get_k_bare_device_core(
    rho_spc_complex, coul_spc_complex, inpv_kpt, phase,
) -> jax.Array:
    """Return batched device c128 bare K; no host transfer."""
```

The host boundary normalizes a single density `(Nk, Nao, Nao)` to
`(1, Nk, Nao, Nao)` and restores that shape convention only after the device
result is returned.  It accepts only fixed-shape c128 floating operands and
validates host-side metadata before dispatch:

- `X.shape == (Nk, Nip, Nao)`, `W.shape == (Nk, Nip, Nip)`, and density shape
  is `(Nk, Nao, Nao)` or `(Nset, Nk, Nao, Nao)`;
- `phase.shape == (Nk, Nk)` and `neg.shape == (Nk,)`; `neg` is an integer,
  in-range involution (`neg[neg[k]] == k`);
- JAX 64-bit mode is enabled and every floating input is c128 (no implicit
  c64 downcast); and
- `exxdiv` is not an argument to the bare device API.

The pure device functions require device operands.  A later explicit host
adapter, not the pure functions, owns these transfers in the initial
no-cache design:

| boundary value | direction | owner and cadence |
| --- | --- | --- |
| PySCF `dm_kpts` | H2D | host adapter, once per `get_jk` call |
| host `inpv_kpt`, `phase`, `neg` | H2D | host adapter, once per call; no retained cache is proposed |
| S4 `coul_kpt` | device resident | passed directly; no D2H materialization |
| preflight ratios | D2H scalar-only | host wrapper, once per device call |
| bare `vk_kpts` | device resident | returned by the pure core |
| PySCF result | D2H array | later adapter only, once when PySCF needs it |

Any retained factor/device-cache owner is out of scope and requires a
separate memory/lifetime policy.  The wrapper may transfer only the stated
scalar diagnostics before the output boundary.  It must not call `np.asarray`
or `jax.device_get` on an array-sized device operand or result.

The jitted implementation is two device stages:

1. A preflight projects each normalized density set and forms
   `rho_spc_complex[s] = P rho[s]` and
   `coul_spc_complex = sqrt(Nk) P W`.  The k transform is explicit on its
   correct axis, for example
   `jnp.einsum("rk,skij->srij", phase, rho)`; it is never applied as
   `phase @ values.reshape(...)` to a batched object with k on axis 1.
   The inverse has the matching form
   `jnp.einsum("rk,srij->skij", phase.conj(), v_spc)`.
2. `neg` is mandatory for the device lane.  With `k=arange(Nk)`, the
   vectorized representative rule is
   `rho_rep[:, k] = rho[:, k]` for `k <= neg[k]`,
   `rho_rep[:, k] = conj(rho[:, neg[k]])` otherwise, followed by
   `rho_rep[:, k] = real(rho_rep[:, k])` when `k == neg[k]`.  The validated
   involution makes each pair use its lower-index representative, exactly
   matching the current loop's visited-pair behavior.
3. Preflight returns one `rho` imaginary ratio per density set and one
   `coul` ratio.  For every transformed value `z`, it uses the exact oracle
   rule: `norm_im / norm_total` when `norm_total > 0`, otherwise `norm_im`.
   The wrapper reads those scalars and enforces `imag_tol` for every set and
   for `coul` before continuing.
4. On success, the module-scope device core consumes the already device-
   resident transforms, takes their real parts, performs the Hadamard
   product, inverse phase transform, and final contraction above.

Keeping preflight separate preserves the NumPy oracle's "validate before
discarding imaginary components" contract rather than silently applying
`jnp.real`.  It also keeps the actual K core free of Python, NumPy, PySCF,
and host transfers.  Both stages use explicit `jnp.einsum`/matrix products
with the supplied unitary phase matrix; they do not reinterpret the canonical
k ordering as an FFT reshape.

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

## Timing contract for a later approved benchmark

This document authorizes no timing.  If a later stage is approved, both
device functions must remain stable module-scope JITs—no function recreation
inside a call—and the first compilation invocation must be reported
separately from warm execution.  Each timed device segment ends with
`.block_until_ready()` (or an equivalent completion barrier).

The report must give raw samples and a median separately for: H2D uploads,
preflight plus scalar synchronization, core execution, the explicit output
D2H transfer, and host Ewald.  The pure bare-K preflight/core timing excludes
array-sized H2D/D2H by construction; the scalar gate synchronization is its
own reported segment.  HBM is measured for the same process/configuration.
No hidden transfer or synchronization may be included in a claimed bare-K
core time.

## Oracle and acceptance gates before any integration

`get_k` remains the independent NumPy oracle.  The full acceptance ladder is
ordered; passing a lower stage authorizes neither a higher stage nor a
default/performance claim.

1. **Private CPU-JAX parity.**  Use identical fixed inputs and require c128
   `atol <= 1e-12` and `rtol <= 1e-12` against `get_k`, with no FFTDF
   substitution for this parity layer.  Cover:

   - Gamma-only real fixture: exact molecular reduction and both single-set and
   batched output shape conventions.
   - A genuine-pair mesh with a valid `neg` map: TR-symmetric Hermitian
   densities, Hermitian output per k point, and device/NumPy agreement at the
   existing machine tier.
   - A near-TR real-SCF-style density: `neg` projects by construction and
   agrees with the NumPy `neg` path; per-set scalar diagnostics are checked
   independently rather than aggregated across a batch.
   - Malformed shapes, non-involutory `neg`, disabled x64, and non-c128 input
   fail at the boundary.  Ewald is rejected by the bare API and stays covered
   by the existing host-oracle tests.
2. **Separately reviewed host-adapter V3 leg.**  Only after private parity and
   review may an adapter transfer the result to PySCF and run the existing
   host Ewald operation.  It must pass K versus FFTDF for both
   `exxdiv=None` and `exxdiv="ewald"`, including genuine-pair/TR and real-SCF
   density fixtures.  The bare signature intentionally has no `exxdiv`.
3. **Separately authorized A100 evidence.**  Only after the adapter V3 gate
   may repeated A100 raw/median/HBM evidence be collected under the timing
   contract above.  CPU parity is not GPU evidence.  No speedup or production
   default claim is permitted until all three stages pass and are reviewed.

Only after the full ladder and review may a separate change decide whether
`ISDFDF.get_jk` selects the device path.  That change must retain the NumPy
oracle test coverage and state its explicit output transfer.
