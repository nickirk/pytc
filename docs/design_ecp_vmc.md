# ECP / Pseudopotential support in pytc VMC — design note

Status: draft, branch `feature/ecp-vmc`.
Scope: VMC only. xTC integration is deferred (see §10).

## 1. Goal

Add semi-local effective core potential (ECP) support to the VMC local-energy
evaluator, driven by ECP data parsed by PySCF (`mol._ecp`). Target use cases
are **ccECP** and **BFD** pseudopotentials, which are the QMC-grade sets.

The deliverable for v1 is a local-energy operator
`Ĥ_ECP ψ / ψ` that replaces `−Z/r` for ECP atoms, evaluated under the
**locality approximation** (trial wavefunction used in the projector
expectation value). DMC-quality fixes (T-moves, determinant locality) are out
of scope.

## 2. ECP form

PySCF stores ECPs as

```
mol._ecp[symbol] = [n_core, [[l, [[r_exp_n, [(zeta, c), ...]]]], ...]]
```

The semi-local potential acting on electron *i* due to ECP atom *A* is

```
V̂_A(r_i) = V_loc^A(r_iA)
           + Σ_{l=0}^{L_A-1} V_l^A(r_iA) P̂_l^A
```

with radial form

```
V_l^A(r) = Σ_k c_{l,k} r^{n_{l,k} - 2} exp(-ζ_{l,k} r²)
```

and angular projector `P̂_l^A = Σ_m |Y_lm^A⟩⟨Y_lm^A|` centered on `R_A`.
The local channel `V_loc` is the `l = -1` entry in PySCF convention (the one
not multiplied by an angular projector).

Effective nuclear charge: `mol.atom_charges()` already returns `Z - n_core`
once an ECP is set, so the `−Z/r` Coulomb tail and ion–ion repulsion are
automatically consistent if we keep using `mol.atom_charges()`.

## 3. Local part — `V_loc(r_iA)`

Trivial multiplicative operator. Implementation in
[hamiltonian.py:72](../pytc/vmc/hamiltonian.py:72), `e_n_potential`:

```
V_en(r_i) = Σ_A [ −Z_eff^A / r_iA  +  has_ecp[A] · V_loc^A(r_iA) ]
```

All-electron atoms have `has_ecp[A] = False` and an empty `V_loc` term — the
existing `−Z/r` branch is unchanged. We mask rather than branch so the kernel
stays vmappable.

Cost: O(N_e · N_atoms · K_loc) per walker, K_loc ≲ 10. Negligible.

## 4. Non-local part — theory

### 4.0 Derivation of the local-energy expression

For each ECP atom $A$ at position $\mathbf{R}_A$ and electron $i$ at position
$\mathbf{r}_i$, write the electron–atom vector
$\mathbf{r}_{iA} = \mathbf{r}_i - \mathbf{R}_A$ with magnitude
$r_{iA} = |\mathbf{r}_{iA}|$ and unit vector
$\hat{\boldsymbol{\Omega}}_i = \mathbf{r}_{iA} / r_{iA}$.

The non-local part of the semi-local ECP operator on electron $i$ is

$$
\hat{V}^A_{\mathrm{NL}}(\mathbf{r}_i)
  = \sum_{l=0}^{L_A-1} V^A_l(r_{iA}) \, \hat{P}^A_l, \qquad
\hat{P}^A_l = \sum_{m=-l}^{l} | Y_{lm}^A \rangle \langle Y_{lm}^A |,
$$

where $Y_{lm}^A$ are spherical harmonics centered on $\mathbf{R}_A$ and acting
only on the angular part of electron $i$ at fixed $r_{iA}$. Inserting
$\langle \mathbf{r}_i' | Y_{lm}^A \rangle = \delta(r_{iA}' - r_{iA}) \,
Y_{lm}(\hat{\boldsymbol{\Omega}}_i')$ gives

$$
\left( \hat{V}^A_{\mathrm{NL}} \psi \right)(\mathbf{r}_i)
 = \sum_{l} V^A_l(r_{iA}) \int d\Omega' \,
   \left[ \sum_m Y_{lm}(\hat{\boldsymbol{\Omega}}_i)\,
                 Y_{lm}^{*}(\hat{\boldsymbol{\Omega}}')\right]
   \psi(\mathbf{r}_i'),
$$

with $\mathbf{r}_i' = \mathbf{R}_A + r_{iA}\,\hat{\boldsymbol{\Omega}}'$
(the displaced electron position; all other electrons fixed). The spherical
harmonic addition theorem

$$
\sum_m Y_{lm}(\hat{\boldsymbol{\Omega}}_i)\,
       Y_{lm}^{*}(\hat{\boldsymbol{\Omega}}')
 = \frac{2l+1}{4\pi}\, P_l\!\left( \hat{\boldsymbol{\Omega}}_i \cdot
                                   \hat{\boldsymbol{\Omega}}' \right)
$$

(with $P_l$ the Legendre polynomial of degree $l$) collapses the $m$-sum and
leaves a single angular integral over the unit sphere $S^2$:

$$
\left( \hat{V}^A_{\mathrm{NL}} \psi \right)(\mathbf{r}_i)
 = \sum_{l} \frac{2l+1}{4\pi}\, V^A_l(r_{iA})
   \int_{S^2} d\Omega'\,
   P_l\!\left(\hat{\boldsymbol{\Omega}}_i \cdot \hat{\boldsymbol{\Omega}}'\right)
   \psi(\mathbf{r}_i').
$$

The local-energy contribution from this electron–atom pair is

$$
\boxed{\;
\frac{ \hat{V}^A_{\mathrm{NL}} \psi }{ \psi }(\mathbf{r}_i)
 = \sum_{l=0}^{L_A-1} \frac{2l+1}{4\pi}\, V^A_l(r_{iA})
   \int_{S^2} d\Omega'\,
   P_l\!\left(\hat{\boldsymbol{\Omega}}_i \cdot \hat{\boldsymbol{\Omega}}'\right)
   \frac{\psi(\mathbf{r}_i')}{\psi(\mathbf{r}_i)} .
\;}
$$

This is exact under the locality approximation, where $\psi$ is the trial
wavefunction $\psi_T$ on both sides. The only electron that moves is
electron $i$; the move is purely angular at fixed radius $r_{iA}$.

### 4.1 Quadrature

The angular integral is replaced by a fixed quadrature on $S^2$,

$$
\int_{S^2} d\Omega'\, f(\hat{\boldsymbol{\Omega}}')
 \approx 4\pi \sum_{q=1}^{N_q} w_q\, f(\hat{\boldsymbol{\Omega}}_q),
\qquad \sum_q w_q = 1,
$$

so the working expression that the kernel evaluates is

$$
\frac{ \hat{V}^A_{\mathrm{NL}} \psi }{ \psi }(\mathbf{r}_i)
 \approx \sum_{l} (2l+1)\, V^A_l(r_{iA})
   \sum_{q=1}^{N_q} w_q\,
   P_l(\cos\theta_q^{iA})\,
   \frac{\psi(\mathbf{r}_{i,q}'^A)}{\psi(\mathbf{r}_i)},
$$

with

$$
\mathbf{r}_{i,q}'^A = \mathbf{R}_A + r_{iA}\,\hat{\boldsymbol{\Omega}}_q,
\qquad
\cos\theta_q^{iA} = \hat{\boldsymbol{\Omega}}_i \cdot \hat{\boldsymbol{\Omega}}_q.
$$

A quadrature of polynomial degree $2L_A$ integrates the projector exactly when
$\psi(\mathbf{r}_i')$ is expanded in spherical harmonics about $\mathbf{R}_A$
truncated at $L_A$. The standard QMC choices are the **12-point icosahedral**
grid (exact through $\ell = 5$) and the **26-point Lebedev** grid (exact
through $\ell = 7$). ccECP and BFD use up to $L = 2$ (d-channel), so 12-point
is enough; 26-point is reserved as a convergence check.

### 4.2 Total ECP energy

The full ECP contribution to the local energy is the sum over all electrons
and all ECP atoms,

$$
E^{\mathrm{ECP}}_L(\mathbf{R})
 = \sum_{i=1}^{N_e} \sum_{A \in \mathcal{A}_{\mathrm{ECP}}}
   \left[
     V^A_{\mathrm{loc}}(r_{iA})
     + \frac{\hat{V}^A_{\mathrm{NL}}\psi}{\psi}(\mathbf{r}_i)
   \right],
$$

and replaces the all-electron $-Z_A / r_{iA}$ term for the same $(i, A)$ pairs.
For non-ECP atoms the bare Coulomb term is kept unchanged.

### 4.3 Wavefunction ratio at displaced position

Per (i, A, q) we need `ψ(R \ r_i ∪ r'_i) / ψ(R)`. Both parts factorize:

```
ratio = (det(Slater(r'_i, others)) / det(Slater(r_i, others))) · exp(J(r'_i) − J(r_i))
```

- **Slater single-electron ratio**: standard fast update — column-i update on
  `inv_up` (if i is α) or `inv_down` (if i is β):
  `ratio_S = Σ_μ φ_μ(r'_i) · inv[:, i]`.
  Already implementable from `walker.inv_up/inv_down` and a recomputed
  φ(r'_i) row from the GTO evaluator. No matrix recomputation, just a
  dot product per electron/quadrature point.
- **Jastrow ratio**: `exp(J(r'_i) − J(r_i))`. For the e–e / e–e–n Jastrow
  forms currently in `pytc/jastrow/`, evaluating `J(r'_i) − J(r_i)` is a
  partial sum over `(i, j)` pairs (j ≠ i) — O(N_e) per displacement. Need a
  helper `jastrow.log_ratio_one_electron(elec_coords, i, new_r_i, params)`
  added to each Jastrow class (or a single generic implementation derived
  from `compute_jastrow_log_value`).

### 4.4 New ansatz API: `psi_ratio_single`

Add to `SlaterJastrow` (and to `SlaterDet`):

```python
def psi_ratio_single(self, walker, i, new_r_i, params) -> jax.Array:
    """ψ(R with electron i moved to new_r_i) / ψ(R). Real scalar (signed)."""
```

This is the workhorse for §4.2. It is also useful elsewhere
(Metropolis one-electron moves currently rebuild matrices in
[pytc/vmc/moves.py](../pytc/vmc/moves.py); a future refactor could use this
ratio path).

For the v1 ECP work we only need `psi_ratio_single` along the displaced-angle
direction; we will not yet rewire Metropolis to use it.

### 4.5 Vectorization plan

Two-level vmap: over electrons (N_e) outermost, over (ECP atoms × quadrature
points) inner. Total cost per walker:

```
N_e · Σ_A∈ECP (n_quad · K_NL^A · (N_basis + N_e))
```

For C/O/N with ccECP: `n_quad = 12`, `K_NL ≈ 4–6`, dominant cost is the GTO
evaluation of φ(r'_i) — `N_e · n_atoms_ecp · n_quad · N_basis` flops, which is
on the same order as the existing Slater build, not catastrophic. Each
displaced point only needs the basis row for one electron, not the full
Slater matrix.

## 4.6 Non-local radius / cutoff

PySCF does **not** expose a user-facing radial cutoff for ECPs. Screening
inside `pyscf/lib/gto/nr_ecp.{c,h}` is exponent-based, via

```c
#define EXPCUTOFF 39      // skip integrand when α·r² > 39 + 6  (≈ e⁻³⁹ ≈ 1e-17)
#define CUTOFF    460     // overflow guard, ~ e⁻⁴⁶⁰
#define ECP_LMAX  5
```

i.e. terms are dropped when the Gaussian overlap exponent makes the
contribution numerically negligible. There is no per-channel `r_max`. For our
QMC use that screening is fine for the ECP **integrals** PySCF builds in
SCF / TC, but it is the wrong primitive for the VMC local-energy evaluator
because we want to **skip electron–atom pairs entirely** when $r_{iA}$ is
large enough that $V_l(r_{iA})$ is negligible — that is, we want a *spatial*
cutoff, not an exponent screen.

The QMC convention (QMCPACK, CASINO) is to define a per-atom non-local
radius $r^A_c$ such that

$$
|V^A_l(r)| < \tau \quad \text{for all } r > r^A_c,\ \text{for all non-local } l,
$$

with tolerance $\tau = 10^{-5}$ Ha (the QMCPACK default, hard-coded as
`const double tolerance = 1.0e-5;` in
`qmcpack/src/QMCHamiltonians/ECPComponentBuilder.2.cpp`; CASINO uses the same
order-of-magnitude convention). Electrons with $r_{iA} > r^A_c$ skip the
non-local evaluation; the local channel $V_{\mathrm{loc}}$ is always added
because it carries the long-range $-Z_{\mathrm{eff}}/r$ tail.

For ccECP the Gaussian exponents are tight enough that typical first-row
radii are

| atom | typical $r_c$ (Bohr) at $\tau = 10^{-5}$ Ha |
|------|---------------------------------------------|
| H    | (no ECP)                                     |
| C    | 1.5 – 2.0                                    |
| N    | 1.5 – 2.0                                    |
| O    | 1.5 – 2.0                                    |
| Si   | 2.5 – 3.0                                    |

These are derived from the smallest Gaussian exponent in each channel of the
ccECP files at <https://pseudopotentiallibrary.org/>. QMCPACK's `Rmax`
default is 2.0 Bohr (`parseCasino` in
`QMCHamiltonians/ECPComponentBuilder.2.cpp`).

**Implementation choice for pytc:** compute $r^A_c$ per atom at
`SlaterJastrow.create` time by scanning $V^A_l(r)$ on a 1-D radial grid until
all non-local channels are below $\tau$. Store as a `(n_atoms,)` array; mask
the non-local contribution per (i, A) by `r_iA < r_c[A]`. This keeps the
kernel branch-free (just multiply by a 0/1 mask) while avoiding the cost of
evaluating $\psi(\mathbf{r}_i')$ at $12 \times n_{\mathrm{ECP\,atoms}}$
quadrature points for every distant electron.

Default tolerance: $\tau = 10^{-5}$ Ha, exposed as an optional kwarg.

## 5. Data layout on the ansatz

Add to `SlaterJastrow` dataclass ([pytc/ansatz/sj.py:11](../pytc/ansatz/sj.py:11)):

```
atom_has_ecp:     (n_atoms,) bool
n_core_per_atom:  (n_atoms,) int            # for diagnostics
ecp_loc:          padded struct of (n_atoms, K_loc_max, 3) for (zeta, c, n_exp)
ecp_nl:           padded struct of (n_atoms, L_max+1, K_nl_max, 3)
ecp_l_mask:       (n_atoms, L_max+1) bool   # which l channels are real
ecp_r_cut:        (n_atoms,) non-local cutoff radius r_c^A   (see §4.6)
quad_dirs:        (n_quad, 3) unit vectors  # shared, single grid
quad_weights:     (n_quad,)
```

Padded layout (zeros where unused) keeps the kernel branch-free and JIT
friendly. Parsing happens once in `SlaterJastrow.create` from `mol._ecp`.

## 6. Cusp considerations

ECPs remove the electron–nucleus Coulomb singularity, so the corresponding
electron–nucleus cusp condition no longer applies for ECP atoms. Two
implications:

- The current Jastrow forms (`REXP`, `BoysHandy`, etc.) may impose or assume
  e–n cusp behavior. We will **leave them as-is for v1** — incorrect cusp
  enforcement at ECP centers is suboptimal but not catastrophic; energy will
  still converge, the Jastrow will just absorb the wrong tail near the
  nucleus.
- Follow-up: add a per-atom flag so e–n Jastrow terms are dropped / softened
  on ECP centers. Tracked as a separate task.

## 7. Ion–ion potential

`mol.atom_charges()` returns Z_eff for ECP atoms, and `SlaterJastrow.create`
already builds `ion_ion_potential` from `atom_charges`
([pytc/ansatz/sj.py:43](../pytc/ansatz/sj.py:43)). No change needed.

## 8. File-by-file change plan

1. **`pytc/ecp/__init__.py`** (new) — module home for ECP code.
2. **`pytc/ecp/parser.py`** (new) — `parse_pyscf_ecp(mol) -> EcpData` returning
   the padded JAX arrays from §5. Pure Python; runs once.
3. **`pytc/ecp/quadrature.py`** (new) — `octahedral_12()` and `lebedev_26()`
   returning `(dirs, weights)`. Static tables.
4. **`pytc/ecp/radial.py`** (new) — `eval_v_loc(ecp_data, r_iA_vec) -> (N_e,)`
   and `eval_v_nl(ecp_data, r_iA_vec) -> (N_e, n_atoms_ecp, L_max+1)`.
   Pure jnp, vmappable.
5. **`pytc/ansatz/sj.py`** — extend dataclass with `ecp` field; parse in
   `create`; add `psi_ratio_single` method.
6. **`pytc/ansatz/det.py`** — add `slater_ratio_single` (single-column
   determinant update via cached inverse).
7. **`pytc/jastrow/*.py`** — add `log_ratio_one_electron` to each Jastrow
   class (or a default in a mixin).
8. **`pytc/vmc/hamiltonian.py`**:
   - Extend `compute_potential_matrix` / `e_n_potential` for V_loc (§3).
   - Add `compute_nonlocal_ecp_energy(sj, walker, params)` (§4) returning the
     scalar contribution to E_L.
   - Add its contribution into `compute_single_walker_energy`.
9. **`pytc/examples/`** — add `c_atom_ccecp_vmc.py` minimal demo.
10. **Tests** under `pytc/tests/` (or wherever the existing test layout is —
    confirm before writing).

## 9. Validation plan

V1 must pass:

1. **Local-only sanity**: set non-local channels to zero by hand → recover
   pyscf HF energy with ECP for J = 0 at converged basis, to within VMC noise.
2. **Single-atom benchmark**: VMC energy of C atom with `ccecp` + simple
   Jastrow vs reference (Bennett et al. 2017 tables) within ~mHa.
3. **Diatomic**: C₂ or N₂ with ccECP, compare to published QMC values.
4. **Locality-approximation invariance**: confirm energy is invariant to
   rotation of the quadrature grid (rotate `quad_dirs` by a random SO(3) and
   re-evaluate — should match to quadrature precision).
5. **Mixed ECP / all-electron**: H–C system with H all-electron, C with
   ccECP. Verifies the masking path.

## 10. Out of scope for v1

- **xTC / TC integral side**: PySCF computes 1e ECP integrals via
  `mol.intor('ECPscalar')`; integration into `pytc/tc.py` and `pytc/xtc.py`
  requires adding ECP commutator terms `[V_loc, J]` and the non-local
  analogue. Separate design pass.
- **DMC / T-moves**: requires sign-resolved ECP move proposals; only matters
  once DMC exists in pytc.
- **Spin-orbit ECPs** (relativistic two-component): out.
- **Jastrow e–n cusp adjustment** at ECP centers (§6): follow-up.
- **Metropolis using `psi_ratio_single`** for one-electron moves: separate
  perf refactor.

## 11. Open questions

1. Numerical floor / cutoff for `r_iA` in `V_loc` and `V_l` to avoid `1/0`
   when an electron coincides with an ECP nucleus. The existing code uses
   `+1e-10`; for ECPs the radial polynomial `r^(n−2)` can amplify this when
   $n_k = 0$ or $1$. Use a small-r series or `jnp.where(r < ε, …, …)`.
   (Resolved separately: non-local **spatial** cutoff $r^A_c$ — see §4.6.)
2. Quadrature rotation per walker / per step (random SO(3))? Reduces angular
   bias for compact basis sets but breaks deterministic E_L. Default: fixed
   grid; revisit if rotation invariance test (§9.4) shows large error.
3. Layout of `ecp_nl` — pad to `L_max` global or per-atom struct-of-arrays?
   Global padding is simpler and fine up to L_max = 2 (ccECP/BFD).
