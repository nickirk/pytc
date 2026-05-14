"""Angular quadrature grids on the unit sphere for ECP non-local evaluation.

Grids are normalized so that sum(weights) == 1 (i.e. the integral
1/(4 pi) * integral_{S^2} f dOmega is approximated as sum_q w_q f(Omega_q)).
This matches the convention used by the working expression in
docs/design_ecp_vmc.md, section 4.1.

Two grids are provided:

* `icosahedral_12`: 12 vertices of a regular icosahedron, equal weights.
  Exact for spherical harmonics up to degree l = 5.
* `lebedev_26`: 26-point cubic-symmetric Lebedev-style grid, exact through
  degree l = 7. Reserved for convergence checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class AngularGrid:
    """Quadrature grid on the unit sphere.

    Attributes:
        directions: (n_quad, 3) array of unit vectors.
        weights: (n_quad,) array of weights, summing to 1.
        max_l_exact: largest l for which the grid integrates Y_lm exactly.
    """

    directions: jnp.ndarray
    weights: jnp.ndarray
    max_l_exact: int

    @property
    def n_points(self) -> int:
        return int(self.directions.shape[0])


def icosahedral_12() -> AngularGrid:
    """12-point icosahedral grid, exact through l = 5.

    Vertices of a regular icosahedron: (0, +-1, +-phi), (+-1, +-phi, 0),
    (+-phi, 0, +-1), with phi = (1 + sqrt(5)) / 2, normalized to unit length.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array(
        [
            [0.0,  1.0,  phi],
            [0.0,  1.0, -phi],
            [0.0, -1.0,  phi],
            [0.0, -1.0, -phi],
            [ 1.0,  phi, 0.0],
            [ 1.0, -phi, 0.0],
            [-1.0,  phi, 0.0],
            [-1.0, -phi, 0.0],
            [ phi, 0.0,  1.0],
            [-phi, 0.0,  1.0],
            [ phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    directions = raw / norms
    weights = np.full(12, 1.0 / 12.0, dtype=np.float64)
    return AngularGrid(
        directions=jnp.asarray(directions),
        weights=jnp.asarray(weights),
        max_l_exact=5,
    )


def lebedev_26() -> AngularGrid:
    """26-point cubic-symmetric Lebedev-style grid, exact through l = 7.

    Three orbits under the cubic group O_h:
        - 6 vertices along +-x, +-y, +-z, each with weight w_a
        - 8 body-diagonal directions (+-1, +-1, +-1)/sqrt(3), each with w_b
        - 12 edge-midpoint directions (+-1, +-1, 0)/sqrt(2) and permutations,
          each with w_c

    Weights from Lebedev (Zh. Vychisl. Mat. Mat. Fiz. 15, 48 (1975))
    normalized to sum to 1:
        w_a = 1/21   (* 6 = 6/21)
        w_b = 9/280  (* 8 = 72/280 = 27/105)
        w_c = 4/105  (* 12 = 48/105)
        Total: 6/21 + 27/105 + 48/105 = 30/105 + 27/105 + 48/105 = 105/105 = 1.
    """
    # Orbit A: 6 axes
    axes = np.array(
        [
            [ 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [ 0.0,  1.0, 0.0],
            [ 0.0, -1.0, 0.0],
            [ 0.0, 0.0,  1.0],
            [ 0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    # Orbit B: 8 body diagonals
    body = (
        np.array(
            [
                [ 1.0,  1.0,  1.0],
                [ 1.0,  1.0, -1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0, -1.0, -1.0],
                [-1.0,  1.0,  1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        / np.sqrt(3.0)
    )
    # Orbit C: 12 edge midpoints (one zero coordinate, other two +-1)
    edges = (
        np.array(
            [
                [ 1.0,  1.0, 0.0],
                [ 1.0, -1.0, 0.0],
                [-1.0,  1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [ 1.0, 0.0,  1.0],
                [ 1.0, 0.0, -1.0],
                [-1.0, 0.0,  1.0],
                [-1.0, 0.0, -1.0],
                [ 0.0,  1.0,  1.0],
                [ 0.0,  1.0, -1.0],
                [ 0.0, -1.0,  1.0],
                [ 0.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        / np.sqrt(2.0)
    )

    directions = np.concatenate([axes, body, edges], axis=0)
    w_a = 1.0 / 21.0
    w_b = 9.0 / 280.0
    w_c = 4.0 / 105.0
    weights = np.concatenate(
        [
            np.full(6, w_a),
            np.full(8, w_b),
            np.full(12, w_c),
        ]
    )
    return AngularGrid(
        directions=jnp.asarray(directions),
        weights=jnp.asarray(weights),
        max_l_exact=7,
    )


_GRIDS = {
    "icosahedral_12": icosahedral_12,
    "lebedev_26": lebedev_26,
}


def get_grid(name: str = "icosahedral_12") -> AngularGrid:
    """Look up a grid by name. Default is the 12-point icosahedral grid."""
    if name not in _GRIDS:
        raise ValueError(
            f"Unknown angular grid '{name}'. Available: {sorted(_GRIDS)}"
        )
    return _GRIDS[name]()
