# benchmarks/

Performance baseline harness for ISDF-XTC-CCSD and VMC kernels.

## Contents

| File | Purpose |
|------|---------|
| `perf_baseline.py` | Timing harness: record timings for canonical systems (H₂O, C₂H₄, benzene), then diff vs a saved baseline to detect regressions. |

## Usage

**Record a new baseline:**
```bash
python benchmarks/perf_baseline.py record --output baseline.json
```

**Compare against a saved baseline:**
```bash
python benchmarks/perf_baseline.py compare --compare baseline.json --threshold 20
```
Exit code is nonzero if any guarded path regresses by more than `--threshold` percent (default 20 %).

## Notes

- A warmup pass is run before timing to pay the JAX JIT compile cost once.
- Timings are medians of `--repeats` (default 5) wall-clock measurements with `jax.block_until_ready` synchronisation.
- The harness runs on CPU (development) or GPU (canonical baseline on the accelerator cluster).
- For `fixed_pivots` correctness coverage, see `pytc/test/test_fixed_pivots.py`.
