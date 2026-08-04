"""FFT-friendly grid selection for periodic builds.

FFT libraries decompose a transform of length ``n`` into stages of small
radices.  Lengths that factor entirely into small primes run through
hand-optimised radix kernels; a length carrying a larger prime factor falls
back to a generic radix-p butterfly, which costs substantially more per grid
point.  cuFFT documents its optimised path as ``2^a 3^b 5^c 7^d``, and DUCC
(the CPU backend behind ``jax.numpy.fft`` and ``scipy.fft``) is structured the
same way.

Plane-wave codes have handled this for decades by choosing the grid rather
than accepting it: Quantum ESPRESSO selects the smallest grid compatible with
the cutoff *and* allowed by the FFT library (``good_fft_order``/``allowed`` in
FFTXlib), and VASP documents the same small-prime rule for NGX/NGXF.  The
direction matters -- they round *up*, so the resulting grid is finer than the
cutoff requires and accuracy never degrades.

Measured cost of ignoring this, at a mesh of 76 = 2^2 x 19 versus its smooth
neighbours (batched 3D complex128 transforms, ns per grid point, relative to
the mean over smooth meshes 64/72/80/84/96):

    device                76 penalty     80^3 vs 76^3 wall time
    V100 / cuFFT   b=8      1.64x               0.703
    V100 / cuFFT   b=64     1.61x               0.715
    CPU  / DUCC    b=8      1.28x               0.844
    CPU  / DUCC    b=64     1.09x               1.005

The effect is large enough on the GPU to invert the usual size ordering: an
80^3 transform carries 17% more data than 76^3 and still completes ~30%
faster.  Pure data scaling would predict 1.17.

Two caveats the table makes visible, both worth knowing before acting on it:

* **The CPU benefit fades with batch size.**  At large batch the CPU working
  set leaves cache entirely and the transform becomes memory-bound, which
  hides the extra radix arithmetic behind memory stalls; by batch 64 the
  penalty is down to 1.09x and 80^3's 17% extra data has cancelled the gain
  outright.  The GPU keeps the penalty across batch because its bandwidth is
  high enough to stay radix-limited.  So this is primarily a *GPU* lever.
* **Smooth does not mean "power of two".**  On CPU, 64 = 2^6 measures 1.34x
  *worse* than the smooth mean -- worse than 76 at batch 64 -- almost
  certainly cache-set aliasing on power-of-two strides.  Rounding up to the
  next allowed size is still right, but do not assume 2^k is optimal.

This module only *reports* a better mesh.  It deliberately does not rewrite
one: a finer grid changes the numbers a calculation produces, so the choice
belongs to the caller who owns the convergence.
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
    """Return the prime factors of ``n`` in ascending order.

    ``factorize(1)`` is the empty tuple, so 1 is trivially friendly to any
    radix set.
    """
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
    """Smallest integer ``>= n`` that factors entirely into ``radices``.

    Rounds up, never down, so a grid chosen this way is at least as fine as
    the one requested and the cutoff it satisfies is still satisfied.
    """
    allowed = _validated_radices(radices)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")
    candidate = n
    # Powers of the smallest radix are always reachable, so this terminates
    # after at most the distance to the next such power.
    while not is_fft_friendly(candidate, allowed):
        candidate += 1
    return candidate


def good_fft_mesh(
    mesh: Sequence[int], radices: Iterable[int] = DEFAULT_FFT_RADICES
) -> tuple[int, ...]:
    """Apply :func:`good_fft_size` to each axis of ``mesh`` independently.

    Axes are independent because a multi-dimensional transform is separable:
    each axis is transformed with its own radix decomposition, so one awkward
    axis penalises only its own passes.
    """
    values = tuple(int(m) for m in mesh)
    if not values:
        raise ValueError("mesh must be a non-empty sequence of positive integers.")
    return tuple(good_fft_size(m, radices) for m in values)


def describe_fft_mesh(
    mesh: Sequence[int], radices: Iterable[int] = DEFAULT_FFT_RADICES
) -> str | None:
    """Return a human-readable diagnostic, or ``None`` when ``mesh`` is fine.

    ``None`` means every axis already factors into ``radices``; there is
    nothing to say and callers should stay silent rather than emit a
    reassuring message on the happy path.
    """
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
        f"{detail}. A large prime factor forces a generic radix butterfly "
        f"instead of an optimised kernel, which can cost ~1.4x per grid point. "
        f"The smallest cutoff-preserving alternative is {suggested} "
        f"({grid_ratio:.2f}x the grid points). Note this trade is not free: a "
        f"finer grid costs every stage that scales with the grid-point count, "
        f"so weigh it against the FFT share of your build before switching."
    )
