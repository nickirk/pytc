# Phase-A factor-direct ISDF VVVV--T2 prototype

This is a standalone validation harness for the existing ISDF K1/K2/K3 and
Delta-U D/X factors.  It has no production call site.  In particular, it does
not route `RCCSD._contract_vvvv_t2` through the prototype, and it does not
alter or approximate the ordinary crossed DF Coulomb term.

## What is checked

`pytc.solver.factor_direct_vvvv` contracts a dense `t2[i,j,c,d]` directly
from the factors in PyTC's raw tile order `V[a,c,b,d]`.

- Full-THC K3 and D use occupied-pair and rank-column panels.
- K1 and K2 use the same contraction independently for each Cartesian
  gradient component, with all three components summed inside one compiled
  executable so the reported XLA accounting is per K1/K2 term rather than a
  single-component proxy.
- X uses its exact partial-THC left- and right-factorized contractions.
- The implementation records direct and pair-swapped K1/K2/K3/D/X branches
  separately.  It applies the current assembly signs exactly:
  `-0.5 * ((K1 - K2 + K3) + pair_swap(...))` and
  `-((D - X) + pair_swap(D - X))`.

The default random-FP64 test checks every individual branch, the assembled
residual, RCCSD `(i,j,a,b) -> (j,i,b,a)` pair symmetry, a structural JAXPR gate
against any `(v,v,v,v)` intermediate, and per-branch steady-state wall time
plus two deliberately distinct memory records:

- a **schedule-intermediate estimate** for the named algebraic panels only;
- the active backend's per-executable **XLA memory analysis** (temporary,
  argument, output, alias, and total bytes).

Neither record is called an allocator high-water bound. In particular, the
prototype uses whole-array `jnp.pad` operations for its small-deck fixed-shape
loops; compiler scratch and those copies are omitted from the schedule estimate
but included in the XLA executable analysis.

## Prototype limitation: X is not yet streamed

The current partial-THC X kernels execute
`x_padded = jnp.pad(x, ...)` before rank-panel slicing.  This keeps both the
input X array and a padded device copy alive. It is acceptable for the H10
small-deck algebra gate, but it is explicitly **not** a production host/disk
streamed-X implementation. Consequently this commit cannot substantiate the
1,200-orbital (or target-GPU) memory feasibility gate by itself; a future,
separately approved production design must stream X panels from host/disk.

## H10 GPU run card

Run this only after the Phase-A code and card are accepted by Felix.  This is a
validation run, not a production solver run and not a task #35/DF--THC run.

```bash
#!/usr/bin/env bash
set -uo pipefail

cd /Users/kl2252/Work/src/pytc-factor-direct
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTC_RUN_FACTOR_DIRECT_H10=1

monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 > factor_direct_h10_gpu_memory.csv &
monitor_pid=$!
set +e
python -m pytest -q -s pytc/solver/test/test_factor_direct_vvvv.py -k H10 \
  2>&1 | tee factor_direct_h10_phase_a.log
test_status=${PIPESTATUS[0]}
set -e
exit "$test_status"
```

Expected evidence:

1. the H10/STO-3G physical ISDF deck passes every direct/pair branch and the
   final residual against `ISDFXTC._assemble_2b_tile`, with relative error no
   larger than `1e-10`;
2. RCCSD pair symmetry passes;
3. the log contains one `FACTOR_DIRECT_TERM` JSON object for each K1/K2/K3/D/X
   direct and pair branch, including synchronized steady-state wall time,
   panel sizes, schedule-intermediate estimate, and compiled XLA temporary /
   argument / output / alias / total bytes;
4. `factor_direct_h10_gpu_memory.csv` is retained as a **process-wide sampled
   trace only**. It spans ISDF setup, test-only dense reference construction,
   and factor-direct work; even with preallocation disabled it is not a
   per-term allocator peak and must not be reported as one.

Record device model, JAX/JAXLIB versions, PyTC commit, and the largest sampled
`memory.used` beside the log, labelled process-wide. If a later task needs an
allocator peak claim, it must add a factor-direct-only baseline/delta or an
isolated one-term runner; this card makes no such claim. A test-only physical
dense tile is allowed because H10/STO-3G is small; the prototype implementation
itself never creates one.
