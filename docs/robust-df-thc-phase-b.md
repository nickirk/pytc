# Phase B: robust DF/THC oracle checkpoint

Status: checkpoint-only design and FP64 numerical oracle.  This document does
not authorize a physical run, a solver call site, a default change, or a
chemistry-tolerance decision.

## Exact PyTC source and pair order

The standard-Coulomb DF source is `pytc/solver/xtc_ccsd.py`:

1. `_init_df_eris` iterates `with_df.loop()` and transforms each auxiliary
   chunk with `_ao2mo.nr_e2(..., aosym='s2', mosym='s1')`.  Thus `Lpq` has
   order `(Q,p,q)` in the active MO basis.
2. It takes `Lpq[:, nocc:, nocc:]`, packs its lower triangle, and writes
   `eris.vvL[(ac),Q]`.
3. The DF paths recover `L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0)`,
   with shape `(a,c,Q)`.  This is the Phase-B source
   `B_(ac),Q = L_vv_full[a,c,Q]`.

The direct on-the-fly VVVV path forms
`np.tensordot(L_vv_full[p], L_vv_full[r], axes=((2,), (2,)))`; its raw tile
is `(a,c,b,d)`.  It transposes to `(a,b,c,d)` only to feed the historical
`einsum('abcd,ijcd->ijab', ...)` interface.  Therefore the algebraic
contraction to preserve is

```text
R_ijab = sum_c,d,Q B_a,c,Q B_b,d,Q t2_ijcd.
```

For an ISDF virtual factor `P[a,mu]`, scalar collocation has precisely the
same C-order flattened pair convention:

```text
C_(ac),mu = P_a,mu P_c,mu,  row(ac) = a * nvir + c.
```

## Oracle algebra

Fit the rectangular DF factor (not the four-index ERI) in FP64:

```text
B_tilde = C W,  W = argmin_W ||C W - B||_F.
J_exact = B B^T
J_LS-THC = B_tilde B_tilde^T
J_robust = B_tilde B^T + B B_tilde^T - B_tilde B_tilde^T.
```

With `DeltaB = B - B_tilde`, the signed identity is

```text
J_exact - J_robust = DeltaB DeltaB^T.
```

The same order holds after a direct VVVV--T2 sandwich:

```text
R_exact - R_robust = sandwich(DeltaB, DeltaB, t2).
```

`pytc/solver/robust_df_thc.py` is the standalone implementation, and
`pytc/solver/test/test_robust_df_thc.py` is its deterministic random-FP64
gate.  Dense pair metrics and test-only VVVV references are allowed inside the
small oracle; none are an implementation plan for the production solver.

## Proposed, not-yet-authorized H10 rank-sweep card

After source/oracle review, the smallest physical validation should reuse the
accepted H10/STO-3G deck and immutable Phase-A base.  It must extract actual
`B[a,c,Q]` through the source path above (DF `loop` → `_ao2mo.nr_e2` →
`vvL` → `unpack_tril`), and actual `P[a,mu]` from the current ISDF object.
It must not manufacture a surrogate B or P.

For the accepted H10 small deck, select actual ISDF columns at ranks
`[4, 8, 12, 16, 20]` (omit values above the available current-ISDF rank).
For each rank, build C from that prefix, fit only W by FP64 least squares,
and emit one JSON record containing the commit, JAX/JAXLIB/backend, H10
dimensions, naux, available ISDF rank, selected rank, Frobenius norms of
`B-B_tilde` and `J_exact-J_robust`, and the direct-sandwich identity error.
The report is an algebra/rank diagnostic only: it must not choose a chemistry
tolerance or claim physical accuracy/speedup.

The future card must retain the Phase-A immutable-worktree guard, explicitly
enable both CPU and CUDA backends (`JAX_PLATFORMS=cuda,cpu`) because current
ISDF setup stages coefficients on CPU, disable JAX preallocation, and label
any `nvidia-smi` trace process-wide/non-allocator/non-production.  It must
first preflight a GPU default plus nonempty GPU/CPU device lists.  No such
card may be submitted without a separate Felix review and explicit dispatch.
