# GPU Memory Management

PyTC auto-sizes GPU tile and panel allocations for the main compute
paths — ISDF K-stream, xTC-CCSD VVVV/vovv/ovvv tiles, and delta-U
direct tiles.
For most workloads you do not need to set anything — the defaults are
designed to use available VRAM safely while leaving headroom for JAX
caches and resident kernels.

## How auto-sizing works

At each relevant sizing point, PyTC combines runtime free-memory and
budget information (from the device or `PYTC_GPU_MAX_MEMORY_MB`) with
an analytical tile-size model, then picks a conservative block size
within a 70% threshold of the available budget. Dispatch-time adaptive
shrink (the delta-U autoshrink path) additionally checks free memory
immediately before the kernel launch and can tighten the block size
further if another process consumed VRAM after setup.

For the CCSD VVVV panel the estimate accounts for the buffers that are
concurrently live during `_assemble_2b_tile`: the tile output, the
`tc_tile` result that is already resident when delta-U is computed, and
the element-wise sum of the two — giving a conservative best-effort
estimate that typically matches what you would set manually.

If another process consumes GPU memory between setup and dispatch (a
genuine OOM), PyTC raises a `RuntimeError` with a message that names
the relevant env-var override so you know exactly what to set.

## Environment-variable overrides

Three variables let you override the auto-sizing.
**All are optional — omit them to get fully automatic behaviour.**

### `PYTC_GPU_MAX_MEMORY_MB`

Authoritative total-GPU budget in MiB.
When set, `adaptive_rank_block_size` treats this as the assumed device
capacity instead of querying XLA.
Use on shared nodes where another user's process holds persistent VRAM
that XLA does not see as "used".

```bash
export PYTC_GPU_MAX_MEMORY_MB=40000  # 40 GB budget on an 80 GB A100
```

### `PYTC_PANEL_BLK`

Hard cap (integer ≥ 1) on the K-stream `panel_size` and
`rank_block_size` used by the ISDF K-integral path.
Use when the ISDF K-stream pre-allocation would exceed available HBM.

```bash
export PYTC_PANEL_BLK=64
```

### `PYTC_SOLVER_BLK`

Hard cap (integer ≥ 1) on the CCSD tile `panel_blk` returned by
`resolve_vvvv_panel_block_sizes` (vvvv path) and
`resolve_v3o_panel_block_size` (vovv/ovvv large-blocks path).
Tile memory scales O(blk²), so halving this value quarters the
per-tile peak.

```bash
export PYTC_SOLVER_BLK=64   # value used for H10/QZ-FNO acceptance runs on an 80 GB A100
```

## When to use the overrides

| Situation | Recommended knob |
|-----------|-----------------|
| Shared node — another tenant holds persistent VRAM | `PYTC_GPU_MAX_MEMORY_MB` |
| ISDF K-stream OOM on a small GPU | `PYTC_PANEL_BLK` |
| CCSD vvvv/vovv OOM on a small GPU or large nkeep | `PYTC_SOLVER_BLK` |
| Avoid shape-recompile churn (you already know the right size) | `PYTC_SOLVER_BLK` or `PYTC_PANEL_BLK` |
| Genuine OOM mid-run (`RuntimeError` citing one of these) | the knob named in the error message |

For a normal run on a dedicated GPU — including large QZ-basis xTC-CCSD
calculations with nkeep up to 300 on an 80 GB A100 — no override is
needed: auto-sizing selects a safe block size automatically.
