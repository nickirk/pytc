# Phase C: fixed-rank robust DF/THC scalable-oracle checkpoint

Status: **test-only source/design and random-FP64 checkpoint**.  This note is
not a production solver route, default, rank selection, tolerance selection,
performance claim, physical calculation, or GPU dispatch.  It deliberately
starts from accepted Phase-B source/oracle commit
`97c1cdaaef26bbd814b2469306496d4efec97d31` on the scratch branch
`scratch/task38-fixed-rank-chemistry`.

## Fixed native-ISDF contract and later chemistry sequence

For every later physical deck, native ISDF must be constructed with

```text
n_rank_phi  = 12 * n_orb
n_rank_grad = 12 * n_orb.
```

The two native pivot sets are fused and de-duplicated by the existing
production-style procedure.  The resulting full fused `P[a,mu]` is the sole
LS-THC factor input.  There is no stored-column prefix, rank sweep, rank
tuning, or legacy `15*n_orb` path in this checkpoint.  Requested rank and
actual fused rank are separate provenance fields: the latter is obtained only
after the native construction and is not assumed to be `24*n_orb`.

The later locked diagnostic order is H10/STO-3G (linear chain, 1.4 Bohr),
H2O/cc-pVDZ (the standard test geometry), N2/cc-pVDZ at 1.10 and 1.80 Å,
Ne/aug-cc-pVQZ, then the production-geometry benzene FNO-200 deck.  REXP uses
`alpha=0.5`.  Those are future physical calculations; none is run here.

The source-dimension preflight gives the following fixed-rank context.  The
comparison is to the real symmetric virtual-pair-space bound
`nvir*(nvir+1)/2`; it is not an effective LS rank or a recommendation.

| future deck | `(norb,nocc,nvir)` | requested density + gradient | nominal fused maximum | symmetric virtual-pair bound |
| --- | ---: | ---: | ---: | ---: |
| H10/STO-3G | `(10,5,5)` | `120 + 120` | `240` | `15` |
| H2O/cc-pVDZ | `(24,5,19)` | `288 + 288` | `576` | `190` |
| N2/cc-pVDZ, both geometries | `(28,7,21)` | `336 + 336` | `672` | `231` |
| Ne/aug-cc-pVQZ | `(80,5,75)` | `960 + 960` | `1,920` | `2,850` |
| benzene FNO-200 target | `(1200,21,1179)` | `14,400 + 14,400` | `28,800` | `695,610` |

Thus the first three small decks are deliberately overcomplete scalar-pair
correctness anchors, while Ne is the first genuinely undercomplete fixed-rank
stress.  The historical benzene fused-count expectation is `25,894`, not a
substitute for a future deck's measured native de-duplication result.

## Test-only panelled algebra

The Phase-B source convention remains metric-applied real-molecular DF
`B[a,c,Q]`, with `B[a,c,Q] = B[c,a,Q]`, and the direct PyTC contraction order
is

```text
R_ijab = sum_c,d,Q B[a,c,Q] B[b,d,Q] t2[i,j,c,d].
```

Let `P[a,m]` have all native fused columns and write `r=n_fused`.  Scalar
pair collocation is only a mathematical definition in this note:

```text
C[(ac),m] = P[a,m] P[c,m].
G = C^T C = (P^T P) elementwise-times (P^T P).
X[m,Q] = (C^T B)[m,Q] = sum_a,c P[a,m] P[c,m] B[a,c,Q].
Y = G^+ X, with the explicit FP64 rcond retained in the diagnostic record.
```

`X` is accumulated over an `a` panel from the metric-applied DF source.  The
implementation holds `P`, `G`, `X/Y`, and source/panel tensors but never
constructs `C[v^2,r]`, a fitted `B_tilde[v^2,Q]`, or `V[a,c,b,d]`.

For a rank panel `(m,n)` and auxiliary panel `Qp`, the contractions are:

```text
T_ijmd = sum_c t2_ijcd P_cm
left-cross_ijab  = sum_m,Q,d P_am Y_mQ B_bdQ T_ijmd
right-cross_ijab = sum_n,Q,c B_acQ Y_nQ (sum_d t2_ijcd P_dn)

Z_ijmn = sum_c,d P_cm t2_ijcd P_dn
M_mn   = sum_Q Y_mQ Y_nQ
full_ijab = sum_m,n P_am Z_ijmn M_mn P_bn

robust_ijab = left-cross_ijab + right-cross_ijab - full_ijab.
```

The independent exact source term is also evaluated in auxiliary panels as
`B_Q @ t2 @ B_Q.T`.  Both partial-THC terms are retained separately in the
test API so the sign and RCCSD pair-swap relation can be audited before their
subtraction.  This is a source/algebra prototype only; it does not change
`pytc/solver/xtc_ccsd.py` or introduce a production call site.

## Deterministic dense-oracle gate

`pytc/solver/test/robust_df_thc_scalable.py` is the standalone panelled helper
and `pytc/solver/test/test_robust_df_thc_scalable.py` is its seeded random
FP64 gate.  The test deliberately builds dense Phase-B oracle tensors only on
a tiny `o=2, v=5, Q=9, r=4` random problem, then verifies panelled `Y`, exact,
both cross terms, full-THC subtraction, and robust result against the dense
oracle.  It proves the signed identity

```text
exact - robust = sandwich(B - B_tilde, B - B_tilde, t2)
```

and verifies the RCCSD pair swap
`left-cross[i,j,a,b] = right-cross[j,i,b,a]` plus pair symmetry of exact,
full, and robust results.  The seeded source arrays are canonical FP64
fingerprinted so the oracle coordinates are explicit:

| input | shape | canonical SHA-256 |
| --- | --- | --- |
| `B` | `[5,5,9]` | `9ac7258759d51b56bae6b56bac36665c9ce73cfabcbcb443cf743abf80eb0fad` |
| `P` | `[5,4]` | `1d5892cb89b32a6d86032548f2b9cea3cb4a1e515a7e29b0d2efeb56fcaa939c` |
| `t2` | `[2,2,5,5]` | `9f51a8cad9c90436913833311146565e5123a00e8f152be9e06acb76c1875286` |

## Shape, FLOP, and memory ledger

Let `o=nocc`, `v=nvir`, `q=naux`, `r=n_fused`, and let `rp`/`qp` be declared
rank/auxiliary panel sizes.  A multiply-add is two FLOPs.  These are the
actual contraction-order leading counts used by the test-only ledger:

| stage | leading FP64 FLOPs |
| --- | --- |
| fit: `P^T P` | `2 v r^2` |
| fit: panelled `C^T B` | `2 v^2 r q` |
| fit: eigensolve / project-back | `r^3 + 4 r^2 q` |
| exact DF sandwich | `4 o^2 v^3 q` |
| both partial-THC cross terms | `8 o^2 r v^2 + 4 o^2 r v^2 q + 4 o^2 r v q` |
| full-THC subtraction | `4 o^2 v^2 r + 4 o^2 v r^2 + 2 r^2 q + o^2 r^2` |

The `phase_c_shape_flop_memory_ledger` helper records every listed tensor in
FP64 elements and bytes, plus conservative live-data estimates.  The latter
are not allocator measurements.  In particular it distinguishes a retained
native `B[v,v,q]` source from the `B[v,v,qp]` panel a later streaming reader
could expose.  The test-only audit API keeps five `t2`-shaped outputs
(exact/two-cross/full/robust) for review; a future implementation need not.

The corresponding conservative live-element estimates are, respectively,
`4r^2 + 2rq` for the fit excluding its immutable source, plus either `v^2q`
(in-core source) or `v^2qp` (streamed source panel); `o^2v^2(1+qp)` for the
exact contraction; `o^2v^2 + o^2rpv + o^2rpvqp` for one partial cross term;
and `o^2v^2 + 2o^2rpv + 2o^2rp^2 + rp^2` for full-THC.  The audit-API peak is
also recorded separately because retaining all five output tensors is a test
diagnostic, not a target implementation policy.

For H10, the prior accepted source has `q=180`; the fixed native request is
`r<=240` until native de-duplication is actually run.  At the nominal value,
the permanent FP64 shapes are `P[5,240]` (9,600 B), `G[240,240]` (460,800 B),
`X/Y[240,180]` (345,600 B), and one output `t2[5,5,5,5]` (5,000 B).  The
forbidden materializations would be `C[25,240]`, `B_tilde[25,180]`, and
`V[5,5,5,5]`; their small H10 size is not a waiver of the no-materialization
rule.

For the target benzene dimensions, use the historical expected `r=25,894`
only as a planning input and leave `q=Q_DF` source-derived until a future
physical deck provides it.  The input/output baseline is nevertheless exact:

| tensor / forbidden object | FP64 elements | size |
| --- | ---: | ---: |
| `P[1179,25894]` | `30,529,026` | 232.918 MiB |
| `G[25894,25894]` | `670,499,236` | 4.996 GiB |
| one `t2[21,21,1179,1179]` output | `613,008,081` | 4.567 GiB |
| forbidden `C[1179^2,25894]` | `35,993,721,654` | 268.174 GiB |
| forbidden fitted `B_tilde[1179^2,Q_DF]` | `1,390,041 * Q_DF` | `10.605 MiB * Q_DF` |
| forbidden `V[1179,1179,1179,1179]` | `1,932,213,981,681` | 14.059 TiB |

For this planning value, the fit-only live estimate excluding B is
`2,681,996,944 + 51,788*Q_DF` elements (19.982 GiB plus 0.395 MiB per
auxiliary function).  This deliberately exposes the Gram/eigensolve-memory
gate before any physical calculation; it is not an allocator or runtime
measurement.

This ledger is a Phase-D input only.  It makes no peak-memory, speedup,
conditioning, rank, chemistry, or integration verdict.  No physical deck or
GPU activity is part of this checkpoint.
