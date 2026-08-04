"""FFT-friendly grid selection for periodic builds.

A length factoring into small primes runs through optimised radix kernels; a
larger prime factor falls back to a generic radix-p butterfly at substantial
cost per grid point. cuFFT's optimised path is ``2^a 3^b 5^c 7^d``; DUCC,
behind ``jax.numpy.fft`` and ``scipy.fft``, is the same. Quantum ESPRESSO and
VASP pick the smallest grid meeting the cutoff that the FFT library allows,
rounding up so the grid is never coarser.

Reports only; does not rewrite a mesh, since a finer grid changes the numbers
a calculation produces.

Limits on the benefit, both measured: the CPU gain vanishes at large batch
(penalty 1.09x at batch 64, where the replacement's extra points cancel it),
and on CPU 64 = 2^6 measures worse than the smooth mean. Numbers and method
in docs/isdf-periodic/artifacts/task92 of the staging repo.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_FFT_RADICES",
    "factorize",
    "is_fft_friendly",
    "good_fft_size",
    "good_fft_mesh",
    "describe_fft_mesh",
]

#: Radices with hand-optimised kernels in both cuFFT and DUCC/pocketfft.
#: Quantum ESPRESSO additionally permits 11; 7 is the conservative common
#: denominator across the backends this package dispatches to.
DEFAULT_FFT_RADICES = (2, 3, 5, 7)


def _validated_radices(radices: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(r) for r in radices)
    if not out:
        raise ValueError("radices must be a non-empty iterable of integers.")
    if any(r < 2 for r in out):
        raise ValueError(f"radices must all be >= 2, got {out}.")
    return out


def factorize(n: int) -> tuple[int, ...]:
    """Return the prime factors of ``n`` in ascending order; ``factorize(1)`` is empty."""
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return tuple(factors)


def is_fft_friendly(n: int, radices: Iterable[int] = DEFAULT_FFT_RADICES) -> bool:
    """True when ``n`` factors entirely into ``radices``."""
    allowed = _validated_radices(radices)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")
    for radix in allowed:
        while n % radix == 0:
            n //= radix
    return n == 1


def good_fft_size(n: int, radices: Iterable[int] = DEFAULT_FFT_RADICES) -> int:
    """Smallest integer ``>= n`` that factors entirely into ``radices``."""
    allowed = _validated_radices(radices)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")
    candidate = n
    while not is_fft_friendly(candidate, allowed):
        candidate += 1
    return candidate


def good_fft_mesh(
    mesh: Sequence[int], radices: Iterable[int] = DEFAULT_FFT_RADICES
) -> tuple[int, ...]:
    """Apply :func:`good_fft_size` per axis; a multi-dim transform is separable."""
    values = tuple(int(m) for m in mesh)
    if not values:
        raise ValueError("mesh must be a non-empty sequence of positive integers.")
    return tuple(good_fft_size(m, radices) for m in values)


def describe_fft_mesh(
    mesh: Sequence[int], radices: Iterable[int] = DEFAULT_FFT_RADICES
) -> str | None:
    """Return a diagnostic, or ``None`` when every axis factors into ``radices``."""
    values = tuple(int(m) for m in mesh)
    if not values:
        raise ValueError("mesh must be a non-empty sequence of positive integers.")
    allowed = _validated_radices(radices)
    hostile = [m for m in values if not is_fft_friendly(m, allowed)]
    if not hostile:
        return None
    suggested = good_fft_mesh(values, allowed)
    detail = ", ".join(
        f"{m} = {' x '.join(str(f) for f in factorize(m))}" for m in sorted(set(hostile))
    )
    grid_ratio = 1.0
    for got, want in zip(values, suggested):
        grid_ratio *= want / got
    return (
        f"FFT grid mesh {values} has axes that do not factor into {allowed}: "
        f"{detail}. Smallest cutoff-preserving alternative: {suggested}, "
        f"at {grid_ratio:.2f}x the grid points -- not free, since every stage "
        f"scaling with grid-point count pays that."
    )
