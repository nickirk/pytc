"""Private subprocess worker for the periodic deterministic CPU solve.

Executed by file path after the parent has narrowed CPU affinity, deliberately
not through ``python -m``: importing the package before setting affinity would
let JAX initialize against the wider allocation and defeat the boundary.
"""

import json
import os
import sys


def _attach_shared_memory(name):
    from multiprocessing import shared_memory

    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13
        block = shared_memory.SharedMemory(name=name)
        # This worker is not the owner. Prevent its independent resource tracker
        # from unlinking the parent's block when the worker exits.
        from multiprocessing import resource_tracker
        resource_tracker.unregister(block._name, "shared_memory")
        return block


def _emit(payload):
    sys.stdout.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def main(metadata_path):
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError(
            f"worker affinity must contain exactly one CPU, got {affinity}."
        )

    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)

    from pytc.pbc.df.isdf import apply_kernel_and_solve_device

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    shape = tuple(int(value) for value in metadata["shape"])
    dtype = np.dtype(metadata["dtype"])
    n_grid = int(metadata["n_grid"])
    pi_shm = _attach_shared_memory(metadata["pi_shm"])
    kern_shm = _attach_shared_memory(metadata["kern_shm"])
    out_shm = _attach_shared_memory(metadata["out_shm"])
    try:
        pi_buf = np.ndarray(shape, dtype=dtype, buffer=pi_shm.buf)
        kern_buf = np.ndarray(shape, dtype=dtype, buffer=kern_shm.buf)
        out_buf = np.ndarray(shape, dtype=dtype, buffer=out_shm.buf)
        # Only the length is consumed when kern_q is precomputed. Reuse one
        # constant vector so the worker does not allocate it once per q.
        phase_stub = np.ones(n_grid, dtype=np.complex128)
        _emit({
            "event": "ready",
            "affinity": affinity,
            "affinity_count": len(affinity),
            "jax_backend": jax.default_backend(),
            "pid": os.getpid(),
        })

        for line in sys.stdin:
            command = json.loads(line)
            if command.get("event") == "stop":
                _emit({"event": "stopped"})
                return
            if command.get("event") != "solve":
                raise RuntimeError(f"unknown worker command: {command!r}")

            q = int(command["q"])
            W_q, projected_kern, info = apply_kernel_and_solve_device(
                None,
                q,
                np.array(pi_buf, copy=True),
                None,
                kern_q=np.array(kern_buf, copy=True),
                phase_q=phase_stub,
                rtol=command["rtol"],
                retained_solve_residual_gate=command[
                    "retained_solve_residual_gate"
                ],
                self_paired=bool(command["self_paired"]),
                retention_mode=command["retention_mode"],
                n_retained_pin=command["n_retained_pin"],
                jitter_rcond=command["jitter_rcond"],
            )
            out_buf[...] = np.asarray(W_q)
            kern_buf[...] = np.asarray(projected_kern)
            info = dict(info)
            info.update({
                "execution_backend": "deterministic_cpu_subprocess",
                "worker_affinity": affinity,
                "worker_affinity_count": len(affinity),
                "worker_jax_backend": jax.default_backend(),
                "worker_pid": os.getpid(),
            })
            _emit({"event": "result", "q": q, "info": info})
    finally:
        pi_shm.close()
        kern_shm.close()
        out_shm.close()


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as exc:
        _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise
