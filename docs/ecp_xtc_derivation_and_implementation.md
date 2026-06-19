# xTC + ECP (pseudopotential) — derivation and implementation

Status: review draft, branch `feature/ecp-vmc`.
Purpose: a self-contained record of the mathematics and the code that
implements it, written so an independent agent can check correctness.
Scope: the **xTC-side** ECP corrections (Phase 1 and Phase 3). The
underlying VMC ECP local-energy operator is documented separately in
`docs/design_ecp_vmc.md`; its §4.0 derivation is the starting point here.

Source files reviewed for this write-up:
- `pytc/ecp/parser.py` — `parse_pyscf_ecp` → `EcpData`
- `pytc/ecp/radial.py` — `eval_v_nl`, `eval_v_loc`, `find_nonlocal_cutoff`
- `pytc/ecp/quadrature.py` — `icosahedral_12`, `lebedev_26`
- `pytc/ecp/energy.py` — `_legendre_p_stack`, VMC non-local energy (reference impl)
- `pytc/jastrow/ncusp.py` — `NuclearCusp`, `eval_chi_single`
- `pytc/xtc_ecp_chi.py` — Phase 1 kernel `compute_delta_h_ecp_chi`
- `pytc/xtc_ecp_du.py` — Phase 3 kernel `compute_delta_U_ecp_du`
- `pytc/xtc.py` — `get_1b_ecp_chi`, `get_2b_ecp_du`, `get_2b`, `make_eris`,
  `_extract_pair_jastrow_fn`, `_find_nuclear_cusp`, `ecp_du_full` plumbing,
  `ISDFXTC.from_xtc`, `_pad_slice_to_panel`
- `pytc/solver/xtc_ccsd.py` — `_make_xtc_eris`

---

## 1. Notation and the bare non-local ECP operator

For ECP atom `A` at `R_A` and electron at `r`, write `r_A = r - R_A`,
`r_{A} = |r_A|`, unit vector `Ω̂ = r_A / r_{A}`. The semi-local ECP is

```
V̂_A(r) = V_loc^A(r_A)  +  Σ_{l=0}^{L_A-1} V_l^A(r_A) P̂_l^A
```

with the radial form (PySCF/GAMESS convention; list index n is the power)

```
V_l^A(r) = Σ_k c_{l,k} · r^{n_{l,k} - 2} · exp(-ζ_{l,k} r²)
```

and angular projector `P̂_l^A = Σ_m |Y_lm^A⟩⟨Y_lm^A|`.

Applying the addition theorem (design note §4.0) collapses the `m`-sum to a
Legendre polynomial, and the **bare** non-local action on a one-electron
function ψ is the angular integral over the sphere of radius `r_A`:

```
(V̂_NL^A ψ)(r) = Σ_l (2l+1)/(4π) V_l^A(r_A) ∫_{S²} dΩ' P_l(Ω̂·Ω̂') ψ(r')
              ≈ Σ_l (2l+1)  V_l^A(r_A) Σ_q w_q P_l(cosθ_q) ψ(r'_q)     (quadrature)
```

with displaced point `r'_q = R_A + r_A Ω_q`, `cosθ_q = Ω̂·Ω_q`, quadrature
weights normalised so `Σ_q w_q = 1` (so the `4π` cancels). This is what
`pytc/ecp/energy.py::compute_nonlocal_ecp_energy` evaluates for VMC and is
the kernel reused by both xTC phases.

**The bare `V_NL` is already in `mf.get_hcore()`** (PySCF builds it via
`ECPscalar`). Everything below is the *correction* the Jastrow dressing
adds on top of that bare term — hence every kernel subtracts the bare
contribution via an `expm1`/`-1`.

---

## 2. Where the corrections come from (transcorrelation of `V_NL`)

xTC works with the similarity-transformed Hamiltonian `H̃ = e^{-J} H e^{J}`,
with Jastrow

```
J(R) = Σ_a χ(r_a)            (1-body, here the NuclearCusp factor)
     + Σ_{a<b} u(r_a, r_b)    (2-body pair Jastrow, e.g. REXP)
```

A **multiplicative local** operator (Coulomb, `V_loc`) commutes with `J`
and is unchanged. The non-local `V_NL` does **not** commute: it displaces
the acted-on electron `i` from `r_i` to `r'_i` (a purely angular move at
fixed radius `r_{iA}`). Therefore the dressed operator picks up the
Jastrow ratio between the displaced and original configurations:

```
e^{-J} V̂_NL,i e^{J}  →   V̂_NL,i  ·  e^{ΔJ_i},
ΔJ_i = J(…,r'_i,…) - J(…,r_i,…)
     = [χ(r'_i) - χ(r_i)]                       ≡ Δχ_i
     + Σ_{j≠i} [u(r'_i, r_j) - u(r_i, r_j)]      ≡ Σ_{j≠i} Δu_{ij}
```

so

```
e^{ΔJ_i} = e^{Δχ_i} · Π_{j≠i} e^{Δu_{ij}}.
```

Expand the pair product and keep only **rank ≤ 2** (linear in the number
of simultaneously-excited pairs — the "honest B'" truncation; products of
two *distinct* `Δu` factors are rank-3+ and dropped):

```
Π_{j≠i} e^{Δu_{ij}}  ≈  1 + Σ_{j≠i} (e^{Δu_{ij}} - 1).
```

Hence the dressed non-local operator, to rank 2, is

```
V̂_NL,i · e^{Δχ_i} · [ 1 + Σ_{j≠i} (e^{Δu_{ij}} - 1) ].
```

This splits into three pieces:

| piece | factor | lands in | name |
|------|--------|----------|------|
| bare | `V_NL · 1` (from `e^{Δχ}=1+…`, the `1`) | `mf.get_hcore()` | (already present) |
| 1-body χ | `V_NL · (e^{Δχ_i} - 1)` | `h1e` | **Phase 1** `Δh^{NL,χ}` |
| 2-body Δu | `V_NL · e^{Δχ_i} · Σ_j (e^{Δu_{ij}} - 1)` | `h2e` | **Phase 3** `ΔU^{NL,Δu}` |

Two consistency checks against the code that the reviewer can confirm:
- Phase 1 uses `jnp.expm1(Δχ)` = `e^{Δχ}-1` (the `-1` removes bare `V_NL`).
- Phase 3 uses the **full** `e^{Δχ}` (`jnp.exp`, not `expm1`) multiplied by
  `jnp.expm1(Δu)` = `e^{Δu}-1`. i.e. the χ-dressing rides along on the
  2-body term too, exactly as the formula above requires.

---

## 3. Phase 1 — 1-body χ resummation `Δh^{NL,χ}_{pq}`

### 3.1 Matrix element

Sandwich `V_NL · (e^{Δχ}-1)` between MOs `φ_p` (bra, at the source point
`r`) and `φ_q` (ket, at the displaced point `r'`). With the spatial
integral over `r` done on the TC grid `{r_g, w_g}`:

```
Δh^{NL,χ}_{pq} = Σ_g w_g φ_p(r_g)
               · Σ_A Σ_l (2l+1) V_l^A(r_{gA})
               · Σ_{q_quad} w_{q} P_l(cosθ_{g,A,q})
               · [ exp(χ(r'_{g,A,q}) - χ(r_g)) - 1 ]
               · φ_q(r'_{g,A,q})
```

with `r'_{g,A,q} = R_A + r_{gA} Ω_q`. Bra `φ_p` is at the **undisplaced**
grid point `r_g`; ket `φ_q` and the χ-numerator are at the **displaced**
point `r'`. `Δχ = χ(r') - χ(r_g)`.

### 3.2 Implementation — `pytc/xtc_ecp_chi.py::compute_delta_h_ecp_chi`

- Geometry built host-side (numpy): `rel[g,A] = r_g - R_A`, `r_gA`,
  `omega_g`; displaced points `displaced[g,A,q] = R_A + r_gA·Ω_q`.
- `φ_q(r')` evaluated host-side via `pyscf.dft.numint.eval_ao` at the
  displaced points, transformed AO→MO by `mo_coeff.T @ ao.T`. Shape
  `phi_disp[orb, g, A, q]`.
- `χ` evaluated by `jax.vmap(chi_fn)` over displaced points and grid points;
  `delta_chi = chi_disp - chi_grid[:,None,None]`; `expm1_chi = expm1(Δχ)`.
- `v_l = eval_v_nl(r_gA, …)`, masked by `has_ecp` on the atom axis.
- `cos_theta = einsum('gad,qd->gaq', omega_g, omega_q)`;
  `P_l = _legendre_p_stack(...)` (3-term recurrence).
- Contractions (index names as in code):
  - `q_integrand[orb,g,A,q] = phi_disp · expm1_chi · w_q`
  - `K[l,orb,g,A] = einsum('lgaq,Qgaq->lQga', P_l, q_integrand)`  (angular sum over q)
  - `angular_kernel[orb,g] = einsum('l,lga,lQga->Qg', (2l+1), v_lga, K)`  (sum over l, A)
  - `delta_h[p,Q] = einsum('pg,Qg->pQ', φ_p·w_g, angular_kernel)`
- Returns zeros if no atom carries an ECP.

### 3.3 Key empirical fact

For **pure-ECP** systems the `NuclearCusp` factor is gated **off** at ECP
atoms (see §6), so `χ ≡ 0` ⇒ `e^{Δχ}-1 = 0` ⇒ `Δh^{NL,χ} = 0` *by
construction*. Phase 1 therefore only matters for **mixed AE+ECP** systems
(e.g. H₂O/BFD, where O is ECP and H is all-electron; `‖Δh‖_F ~ 1 µHa`).
This is expected, not a bug — it is why Phase 3 was needed to move
pure-ECP benchmarks.

---

## 4. Phase 3 — 2-body honest-B' `ΔU^{NL,Δu}_{pqrs}`

### 4.1 Operator and matrix element

The rank-2 piece is a genuine 2-body operator: `V_NL` acts on electron 1
(displacing `r_1 → r'_1`), weighted by `e^{Δχ_1}·(e^{Δu_{12}}-1)`, with
electron 2 a **spectator** (its coordinate `r_2` enters only through
`u(·, r_2)`; it is *not* displaced and *not* density-contracted). In
chemist's notation `(pq|rs)` — `(p,q)` = electron-1 bra/ket, `(r,s)` =
electron-2 bra/ket:

```
ΔU^{NL,Δu}_{pqrs} = Σ_{r1} Σ_{r2} w_{r1} w_{r2}
                    φ_p(r1) · angK_q(r1, r2) · φ_r(r2) φ_s(r2)

angK_q(r1,r2) = Σ_A Σ_l (2l+1) V_l^A(r_{1A})
              · Σ_{q_quad} w_q P_l(cosθ) · e^{Δχ_{1,A,q}}
              · ( exp[ u(r'_{1,A,q}, r2) - u(r1, r2) ] - 1 )
              · φ_q(r'_{1,A,q})
```

This mirrors how xTC already treats the kinetic-Jastrow 2-body piece
(`get_delta_U` / `kmat.calc_K3`): `r_2` is a free grid axis and the ket
side contracts against `φ_r(r2)φ_s(r2)w_{r2}`. **No HF-density projection
is imposed** (an earlier draft did a Hartree-only projection of ΔU; it was
wrong-signed by ~0.33 mHa and was removed — commit `2f4e65e`).

### 4.2 Symmetrisation

The kernel as written places `V_NL` on electron 1. The physical pair
operator must be symmetric under electron 1 ↔ 2 exchange, so the
`V_NL`-on-electron-2 partner is added by transposing the `(p,q)` and
`(r,s)` index pairs:

```
ΔU ← ΔU + transpose(ΔU, (2,3,0,1))      # symmetrize=True (default)
```

**[Verify]** that this `(pq)↔(rs)` symmetrisation is the correct
accounting for "V_NL acts on either electron of the pair" and introduces
no double-count / missing factor of 2 relative to the `Σ_{i} Σ_{j≠i}`
sum in §2 and the second-quantised `ΔU_{pqrs} a†_p a†_r a_s a_q`
normalisation expected by the downstream RCCSD ERIs.

### 4.3 Implementation — `pytc/xtc_ecp_du.py::compute_delta_U_ecp_du`

Mirrors `kmat.calc_K3`'s nested-scan layout:
- **outer scan over r₂ batches** carrying `φ_r(r2), φ_s(r2), w_{r2}`;
- **inner scan over r₁ batches** carrying `φ_p(r1), w_{r1}`, the displaced
  positions `r'_{1,m}` (flattened over `m=(A,q_quad)`), and a precomputed
  prefactor.

Precomputed, r₂-independent angular prefactor:
```
alpha[g,A,q]      = Σ_l (2l+1) V_l(g,A) P_l(g,A,q)        # einsum 'l,lga,lgaq->gaq'
alpha            *= w_q                                    # absorb angular weights
exp_dchi[g,A,q]   = exp(χ(r'_{g,A,q}) - χ(r_g))           # full exp, or 1 if chi_fn is None
angK_prefactor    = alpha · exp_dchi
phi_disp_pref[orb,g,m] = φ_q(r'_{g,m}) · angK_prefactor   # m = (A,q_quad)
```
Inner-scan body, per `(r1 batch i, r2 batch j)`:
```
u_r1r2[i,j]   = u(r1_i, r2_j)                              # pair_fn on Cartesian product
u_disp[i,m,j] = u(r'_{i,m}, r2_j)
Du[i,m,j]     = u_disp - u_r1r2[:,None,:]
E[i,m,j]      = expm1(Du)
angK[q,i,j]   = einsum('qim,imj->qij', pref_batch, E)
contrib[p,q,j]= einsum('pi,qij->pqj', φ_p·w1, angK)       # accumulate over r1
```
Outer accumulation: `einsum('pqj,rj,sj->pqrs', tmp, φ_rs·w2, φ_rs)`.
Grids are padded to multiples of `inner_batch`/`outer_batch` (pad value 0,
so padded rows contribute nothing); `jax.checkpoint` wraps both scans.

`pair_fn = u(r1,r2)` is assembled by `XTC._extract_pair_jastrow_fn`: it
sums `_compute(r1,r2,p)` over every **non-NuclearCusp** Jastrow factor
(NuclearCusp is the χ piece, handled separately). Returns `None` (⇒ Phase 3
= 0) if there is no pair factor. `chi_fn` comes from `_find_nuclear_cusp`
and is `None` if no NuclearCusp ⇒ `e^{Δχ}` replaced by 1.

### 4.4 Empirical anchor

Be/ccECP-VDZ, REXP α=0.5: `‖ΔU^{NL,Δu}‖_F = 40.8 mHa`;
`E(xtc-CCSD, Phase 1+3) = -1.01939 Ha`; shift vs Phase-1-alone = **-1.04
mHa**, variational/correct direction.

---

## 5. Integration into the solver

### 5.1 Full-tensor path — `XTC.make_eris` (`pytc/xtc.py`)

1. If `use_ecp_du`: `ecp_du_full = get_2b_ecp_du(mf, jastrow_params)`
   (a host numpy rank-4 tensor) is **bound onto a cloned XTC** via
   `self.replace(ecp_du_full=...)`. `ecp_du_full` is a non-pytree field.
2. `get_const`, `get_1b`, `get_2b` are then called on the clone. `get_2b`
   (§5.2) automatically adds the `ecp_du_full` slice — so **no separate
   `h2e_ecp_du` addition is written**; the Fock build, the all-occupied
   blocks, the ao2mo `vvvv` writer, etc. all pick it up implicitly.
3. If `use_ecp_chi`: `h1e_ecp_chi = get_1b_ecp_chi(mf, jastrow_params)` is
   added to `h1e = h1e_std + h1e_corr + h1e_ecp_chi`.
4. `h2e = eri_std + h2e_corr`; Fock = `h1e + (2·J - K)` over occupied; ERI
   blocks sliced out into the `_ChemistsERIs` object for RCCSD.

### 5.2 `XTC.get_2b` — the single injection point

Returns `TC + Δu(kinetic) + (ecp_du_full slice if set)`. The ECP-Δu add is
a pure host-side numpy `+=` (no JAX), gated on `self.ecp_du_full is not
None`, sliced by `ranges` when a sub-block is requested.

### 5.3 Streamed DF/ISDF path

For production-size systems xTC streams ERI blocks
(`_compute_large_blocks`, `_compute_medium_blocks_tiled`,
`_compute_vvvv_block_df`, `_compute_vvvv_block_ao2mo`) rather than building
the full tensor. These call `XTC.get_2b(ranges=...)` /
`ISDFXTC._assemble_2b_tile(...)`, both of which add the matching
`ecp_du_full[ranges]` slice. `_pad_slice_to_panel` pads the numpy slice to
the JIT-stable panel layout (`pr`/`qr`/`ps`) so the ECP-Δu slice lines up
with the padded TC+Δu tile.

**Bug fixed in commit `afd7c83`:** `ISDFXTC.from_xtc` was silently dropping
`xtc_obj.ecp_data`, so any run on the streamed path got **zero** from both
Phase 1 and Phase 3. Now `ecp_data` is forwarded into the cloned ISDFXTC.
**[Verify]** the forwarding is complete and that `ecp_du_full` (bound by
`make_eris`) also survives any cloning on the streamed path.

### 5.4 `pytc/solver/xtc_ccsd.py::_make_xtc_eris`

Simplified to bind `ecp_du_full` once and rely on the implicit `get_2b`
add (removed the explicit Fock-build add and per-block slice adds).

---

## 6. NuclearCusp / χ and the 1/(N−1) factor

`χ(r)` comes from `NuclearCusp.eval_chi_single(r, params)`
(`pytc/jastrow/ncusp.py`). NuclearCusp enforces the Kato e–n cusp; at ECP
atoms there is no Coulomb singularity, so the per-atom contribution is
gated off (`has_ecp_per_atom` mask inside `_compute_inner`, returning 0 for
ECP centers). This is the standard QMC convention (QMCPACK/CASINO/QWalk).

**Critical factor [Verify]:** `_compute_inner` divides the per-electron sum
by `(n_e − 1)` to match CASINO's TERM-convention 1/(N−1) scaling used by
the pair-Jastrow accumulator. `eval_chi_single` **undoes** this by
multiplying by `(self.nelectron − 1)` to recover the *natural* per-electron
χ(r). The xTC χ-resummation needs the natural value, so this un-scaling
must be present and correct — this is exactly the class of factor bug seen
before in DTN debugging. Confirm the `(N−1)` cancels to give the physical
single-electron χ and that no double-application occurs when χ is also used
inside Phase 3's `exp(Δχ)`.

---

## 7. Assumptions, approximations, and explicit things to verify

1. **Rank-2 truncation ("honest B'")**: drops products of two distinct
   `Δu` factors (rank-3+). This is the central approximation; its size is
   not bounded in-code. Worth a numerical probe (e.g. compare to a
   higher-rank reference on a 2-electron system where the product term is
   absent so the truncation is exact — Be has 2 valence electrons under
   ccECP, which is why it was the test case).
2. **Locality approximation** of `V_NL` (trial wavefunction on both sides)
   is inherited from the VMC operator; xTC's "trial" is the Jastrow
   dressing. Standard, but non-variational in general.
3. **Angular quadrature** is icosahedral-12 (exact through ℓ=5). ccECP/BFD
   use up to ℓ=2, so the bare projector is integrated exactly; but the
   dressing `e^{Δχ}(e^{Δu}-1)φ_q(r')` is **not** a low-degree polynomial in
   Ω, so the quadrature error on the *correction* is uncontrolled.
   **[Verify]** convergence vs `lebedev_26` (exact through ℓ=7) — the
   rotation-invariance / grid-convergence test from design §9.4 applied to
   `Δh`/`ΔU`, not just the bare energy.
4. **Symmetrisation factor** (§4.2) — double-count / factor-of-2 check
   against the second-quantised normalisation.
5. **`(N−1)` χ-scaling** (§6).
6. **Radial `r^{n-2}` near r→0**: `eval_radial_channel` floors `r` at
   `1e-8`; for `n_k ∈ {0,1}` the power is negative. Grid points very close
   to an ECP nucleus could amplify noise. **[Verify]** the TC grid never
   places weight pathologically close to an ECP center, or that the floor
   is harmless there.
7. **Memory of `get_2b_ecp_du(ranges=...)`**: builds the **full** `n_orb⁴`
   tensor then slices — O(n_orb⁴) memory even for a small requested block.
   Fine for small systems; flagged as future work for the streamed path.
8. **Bra/ket point assignment** in both kernels (`φ_p` at undisplaced `r_g`,
   `φ_q` at displaced `r'`; `Δχ = χ(r')−χ(r_g)`): confirm this matches the
   `⟨φ_p| V_NL · g |φ_q⟩` operator ordering (V_NL displaces the *ket*
   coordinate).
9. **Hermiticity**: `make_eris` feeds these into PySCF `RCCSD` ERIs which
   assume specific symmetry. Phase 1 `Δh` is built without explicit
   symmetrisation — confirm it is symmetric (or is symmetrised downstream).
   Phase 3 is `(pq)↔(rs)`-symmetrised but confirm the 8-fold ERI symmetry
   assumptions of `_ChemistsERIs` are not silently violated (the operator
   is genuinely non-symmetric in `p↔q` because `V_NL` is non-local).

---

## 8. Test surface already present

- `pytc/test/test_xtc_ecp_chi.py` — 6 tests: no-ECP (zero), no-NuclearCusp
  (zero), constant-χ, Hermitian symmetry, Be/ccECP-VDZ smoke (=0 by
  design), H₂O/BFD-VDZ smoke.
- `pytc/test/test_xtc_ecp_du.py` — rank-4 API, `(pq)↔(rs)` symmetry, and
  `test_kernel_matches_independent_integration` (hand-built integrand on
  `grid_lvl=0` vs production scan, 1e-6 rel / 1e-12 abs, REXP-only so χ is
  absent and the 2-body kernel is isolated).
- `pytc/test/test_xtc_ecp_du_streamed.py` — block-by-block parity between
  streamed and full-tensor paths across all 13 ERI blocks (atol/rtol 1e-9);
  `ranges=` test on `get_2b_ecp_du`.
- `pytc/ecp/test/` — parser, quadrature, radial, high-ℓ non-local, overlap
  warning, integration.
- `pytc/jastrow/test/test_ncusp_ecp.py` — cusp gating at ECP atoms.

**Gaps the reviewer may want new tests for** (map to §7):
quadrature-convergence of the *correction* (icosa-12 vs lebedev-26);
rank-2 truncation error on a 2-electron reference; an independent-integration
check for **Phase 1** analogous to the Phase 3 one (the existing Phase 3
test runs REXP-only, so χ-dressing in Phase 3 is never exercised against
an independent integral); the `(N−1)` χ-scaling pinned by a direct
numerical comparison of `eval_chi_single` against a hand-evaluated χ.
