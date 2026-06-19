# Task #10 — FNO/MP2-NO validation (`make_fno_mo_coeff`)

`make_fno_mo_coeff(mf, n_keep=None, occ_threshold=None)` builds MP2 natural
orbitals, keeps the top-`n_keep` virtuals by occupation (or those
`>= occ_threshold`), semicanonicalizes the kept-virtual Fock block, and
returns a **truncated `mf` copy** (+ matching `mo_coeff`/`mo_occ`/
`mo_energy`) that drops into `ISDFXTC.from_pyscf(mf_fno, ...) -> xTC-CCSD`
with `n_orb < n_ao`. Design B: no kernel/API change (TC.from_pyscf sets
`n_orb = mo_coeff.shape[1]`; XTC.from_pyscf + solver read mo_coeff/mo_occ
from the single truncated mf).

Reproduce:
```
PYTHONPATH=. python pytc/test/validate_fno_xtc_ccsd.py 10 cc-pvdz 0 10,25,45 0.5 isdf
```

## Two bugs caught while wiring it (both fixed, neither an API change)
1. MOs are orthonormal in the **AO-overlap** metric `C^T S C = I`, not the
   Euclidean one. The MP2 1-RDM must be transformed with the metric:
   `gamma_mo = C^T S P_ao S C` (a plain `C^T P C` is wrong and breaks NO
   orthonormality by ~O(10)).
2. `copy.copy(mf)` trips pyscf `__getstate__`, which nulls the cached
   `mf._eri` and breaks the solver's `ao2mo`. Use `mf.copy()`.

## n_mo == n_ao assumptions (none force an API change)
- `TC.from_pyscf`: `n_orb = mo_coeff.shape[1]` (SAFE).
- `XTC.from_pyscf`: stores `mo_occ = mf.mo_occ` (consistent — truncated mf
  has `len(mo_occ) == n_orb_fno`).
- `_get_mf_dm`: `jnp.diag(self.mo_occ)/2` (reads truncated mo_occ, SAFE).
- solver `_make_xtc_eris`: `make_rdm1(mo_coeff, mo_occ=cc._scf.mo_occ)`
  (both from the single truncated mf, SAFE).
- ISDF rank: `from_xtc` default `n_rank = n_grid//4` (grid-derived,
  MO-indifferent — over-provisioned on truncation but correct). The
  `12*n_orb` factor used in the solver tests is a caller choice and DOES
  scale with the truncated `n_orb`. Pivots come from `isdf_decompose(phi,
  ...)`, so they adapt to the truncated MOs (no stale full-MO pivots).

## Operational note for Grace (task #8, H-chain/5Z scan)
- `ISDFXTC.from_xtc(...).isdf(jastrow_params)` **preload the kernels** —
  without it `ao2mo` recomputes delta_U kernels on-the-fly and is slow.
- Each `n_keep` gives a different `mo_coeff`, so do NOT reuse an ISDF
  `save_path` cache across truncations — `from_xtc`'s gauge check will
  reject the mismatch. Use a fresh cache path (or `save_path=None`) per
  `n_keep`.
- `make_fno_mo_coeff` also accepts `occ_threshold` for ke-liao's
  threshold scan (keeps every virtual with NO occupation `>= threshold`).

## Results — H4/cc-pvdz (exact XTC, grid_lvl=0, alpha=0.5)
```
 n_keep  n_orb        E_tot      dE vs full
      2      4   -2.1460782487   +1.05e-02
      4      6   -2.1509708753   +5.66e-03
     18     20   -2.1566268803   -3.91e-14   <- all-virtual == full (numerically exact)
```

## Results — H4/cc-pvdz (ISDF production path, n_rank=12*n_orb, grid_lvl=0)
```
 n_keep  n_orb        E_tot      dE vs full
   full     20   -2.1566268195        —
      2      4   -2.1460782487   +1.05e-02
      4      6   -2.1509708752   +5.66e-03
     18     20   -2.1566268325   -1.30e-08   <- all-virtual == full (ISDF gauge)
```
FNO energies are identical to the exact-XTC column (ISDF saturated at
n_rank = 12*n_orb), cross-validating both paths.

## Results — H10/cc-pvdz (ISDF production path, n_rank=12*n_orb, grid_lvl=0, alpha=0.5)
```
 n_keep  n_orb        E_tot      dE vs full   n_orb<n_ao
   full     50   -5.4015936004        —           —
     10     15   -5.3808450551   +2.07e-02        OK
     25     30   -5.4011481077   +4.45e-04        OK   (sub-mHa at 67% of virtuals)
     45     50   -5.4015938770   -2.77e-07        FULL (all-virtual == full)
```

Checks (Felix deliverable #2): (a) `n_orb < n_ao` runs (15, 30 < 50) ✓;
(b) nocc/mo_occ stay consistent (unit tests + runs) ✓; (c) `n_keep -> nvir`
reproduces the full energy to 2.8e-7 through the ISDF production path ✓.

grid_lvl=0 is used here for CPU turnaround on the work machine; FNO
convergence behaviour is separable from grid error, so it does not affect
the method validation. Production H10/cc-pV5Z (Grace, GPU) should use the
production grid level.
