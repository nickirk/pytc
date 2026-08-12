"""Batched three-dimensional FFT throughput and memory benchmark.

Run one case per fresh process so peak-resident memory belongs to one batch
shape. The production screen uses a 57x57x57 mesh and treats the leading axes
as an interpolation-point panel and a channel batch.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


K1_FORWARD_TRANSFORMS = 2_552_580
K1_INVERSE_TRANSFORMS = 10_460_880
K1_TOTAL_TRANSFORMS = K1_FORWARD_TRANSFORMS + K1_INVERSE_TRANSFORMS


@dataclass(frozen=True)
class FFTBenchmarkCase:
    mesh: tuple[int, int, int] = (57, 57, 57)
    panel_size: int = 1
    channel_batch: int = 1
    repeats: int = 5
    seed: int = 9182

    def __post_init__(self) -> None:
        mesh = tuple(int(value) for value in self.mesh)
        if len(mesh) != 3 or any(value <= 0 for value in mesh):
            raise ValueError("mesh must contain three positive integers")
        object.__setattr__(self, "mesh", mesh)
        for name in ("panel_size", "channel_batch", "repeats"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def batch_size(self) -> int:
        return self.panel_size * self.channel_batch

    @property
    def array_shape(self) -> tuple[int, ...]:
        return (self.panel_size, self.channel_batch, *self.mesh)


def fft_flops(mesh: tuple[int, int, int]) -> float:
    """Return the conventional 5 N log2(N) complex-FFT planning count."""
    n_grid = math.prod(mesh)
    return 5.0 * n_grid * math.log2(n_grid)


def projected_wall_seconds(transform_count: int, transforms_per_second: float) -> float:
    if transform_count < 0:
        raise ValueError("transform_count must be nonnegative")
    if not math.isfinite(transforms_per_second) or transforms_per_second <= 0.0:
        raise ValueError("transforms_per_second must be finite and positive")
    return transform_count / transforms_per_second


def projected_k1_wall_seconds(
    forward_transforms_per_second: float,
    inverse_transforms_per_second: float,
) -> dict[str, float]:
    """Project the K1 transform wall using direction-specific warm rates."""
    forward_seconds = projected_wall_seconds(
        K1_FORWARD_TRANSFORMS, forward_transforms_per_second
    )
    inverse_seconds = projected_wall_seconds(
        K1_INVERSE_TRANSFORMS, inverse_transforms_per_second
    )
    total_seconds = forward_seconds + inverse_seconds
    return {
        "forward_seconds": forward_seconds,
        "inverse_seconds": inverse_seconds,
        "total_seconds": total_seconds,
        "effective_transforms_per_second": K1_TOTAL_TRANSFORMS / total_seconds,
    }


def recommended_cases() -> list[FFTBenchmarkCase]:
    return [
        FFTBenchmarkCase(panel_size=1, channel_batch=1),
        FFTBenchmarkCase(panel_size=1, channel_batch=8),
        FFTBenchmarkCase(panel_size=4, channel_batch=8),
        FFTBenchmarkCase(panel_size=8, channel_batch=16),
        FFTBenchmarkCase(panel_size=16, channel_batch=32),
    ]


def _sync(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def _host_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _device_memory() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    memory_stats = {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, float))
        and ("byte" in key.lower() or "peak" in key.lower() or "limit" in key.lower())
    }
    return {
        "device": str(device),
        "platform": device.platform,
        "memory_stats": memory_stats,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _time_transform(transform: Any, device_input: Any, repeats: int) -> dict[str, Any]:
    start = time.perf_counter()
    output = transform(device_input)
    _sync(output)
    compile_and_first_seconds = time.perf_counter() - start

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = transform(device_input)
        _sync(output)
        samples.append(time.perf_counter() - start)

    return {
        "output": output,
        "compile_and_first_seconds": compile_and_first_seconds,
        "warm_seconds": [float(value) for value in samples],
        "warm_median_seconds": float(statistics.median(samples)),
    }


def run_case(case: FFTBenchmarkCase) -> dict[str, Any]:
    rng = np.random.default_rng(case.seed)
    real = rng.standard_normal(case.array_shape)
    imag = rng.standard_normal(case.array_shape)
    host_input = np.asarray(real + 1j * imag, dtype=np.complex128)
    del real, imag

    start = time.perf_counter()
    device_input = jax.device_put(host_input)
    _sync(device_input)
    host_to_device_seconds = time.perf_counter() - start

    forward = _time_transform(
        jax.jit(lambda values: jnp.fft.fftn(values, axes=(-3, -2, -1))),
        device_input,
        case.repeats,
    )
    inverse = _time_transform(
        jax.jit(lambda values: jnp.fft.ifftn(values, axes=(-3, -2, -1))),
        device_input,
        case.repeats,
    )

    start = time.perf_counter()
    host_output = np.asarray(jax.device_get(inverse["output"]))
    device_to_host_seconds = time.perf_counter() - start

    flops_per_transform = fft_flops(case.mesh)
    for timing in (forward, inverse):
        timing["transforms_per_second"] = (
            case.batch_size / timing["warm_median_seconds"]
        )
        timing["achieved_gflops"] = (
            timing["transforms_per_second"] * flops_per_transform / 1e9
        )
        del timing["output"]
    k1_wall = projected_k1_wall_seconds(
        forward["transforms_per_second"], inverse["transforms_per_second"]
    )
    input_bytes = int(host_input.nbytes)
    output_bytes = int(host_output.nbytes)

    return {
        "schema": "pytc.pbc.fft_benchmark.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "case": asdict(case),
        "batch_size": case.batch_size,
        "array_shape": case.array_shape,
        "dtype": str(host_input.dtype),
        "n_grid": math.prod(case.mesh),
        "flops_per_transform": flops_per_transform,
        "forward": forward,
        "inverse": inverse,
        "host_to_device_seconds": host_to_device_seconds,
        "device_to_host_seconds": device_to_host_seconds,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "structural_input_output_bytes": input_bytes + output_bytes,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
        "device": _device_memory(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "k1_projection": {
            "forward_transform_count": K1_FORWARD_TRANSFORMS,
            "inverse_transform_count": K1_INVERSE_TRANSFORMS,
            "total_transform_count": K1_TOTAL_TRANSFORMS,
            **k1_wall,
        },
    }


def _write_result(result: Any, output: str | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", nargs=3, type=int, default=(57, 57, 57))
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--channel-batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9182)
    parser.add_argument("--output")
    parser.add_argument("--emit-matrix", action="store_true")
    args = parser.parse_args(argv)

    if args.emit_matrix:
        _write_result([asdict(case) for case in recommended_cases()], args.output)
        return 0

    case = FFTBenchmarkCase(
        mesh=tuple(args.mesh),
        panel_size=args.panel_size,
        channel_batch=args.channel_batch,
        repeats=args.repeats,
        seed=args.seed,
    )
    _write_result(run_case(case), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
