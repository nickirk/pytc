"""FFT-backed Gamma-point periodic TC and xTC quadrature.

This module provides a correctness-first FFT backend on a uniform cell grid.
It deliberately supports a general periodic Jastrow, including the
centre-dependent Boys--Handy form: for each left grid point it represents the
right-coordinate kernel as a circular-convolution kernel and evaluates the
required potential with FFTs.  The resulting algorithm is not yet the
low-rank/channel-compressed production algorithm, but it is an independent
implementation seam on which that optimisation can be built without changing
the integral algebra.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import jax.numpy as jnp
import numpy as np
from flax import struct
from pyscf.pbc import dft as periodic_dft

from pytc.integrals.tc import TC
from pytc.integrals.xtc import XTC

from .tc import _require_gamma_real


def _mesh_tuple(mesh) -> tuple[int, int, int]:
    result = tuple(int(value) for value in mesh)
    if len(result) != 3 or any(value < 1 for value in result):
        raise ValueError(f"FFT mesh must contain three positive integers, got {mesh!r}")
    return result


def _validate_uniform_grid(grid_points, weights, mesh):
    mesh = _mesh_tuple(mesh)
    n_grid = reduce(mul, mesh, 1)
    if np.shape(grid_points) != (n_grid, 3):
        raise ValueError(
            f"FFT grid shape must be ({n_grid}, 3) for mesh {mesh}, "
            f"got {np.shape(grid_points)}"
        )
    weights_np = np.asarray(weights)
    if weights_np.shape != (n_grid,):
        raise ValueError(f"FFT weights must have shape ({n_grid},)")
    if not np.allclose(weights_np, weights_np[0], rtol=1e-12, atol=1e-14):
        raise ValueError("FFT TC requires a uniform periodic quadrature")
    return mesh


def _source_indices(left_flat: int, mesh: tuple[int, int, int]):
    """Return y indices ordered by displacement s=x-y on a periodic mesh."""
    left = np.unravel_index(left_flat, mesh)
    axes = [
        (left[axis] - np.arange(mesh[axis], dtype=np.int64)) % mesh[axis]
        for axis in range(3)
    ]
    yy = np.meshgrid(*axes, indexing="ij")
    return np.ravel_multi_index(tuple(yy), mesh).reshape(-1)


def fft_pair_potential(
    grid_points,
    weights,
    mesh,
    jastrow_factor,
    jastrow_params,
    right_functions,
    *,
    squared_gradient=False,
):
    r"""Apply the periodic pair kernel to right-grid functions with FFTs.

    For every left grid point ``x`` this evaluates

    ``V_a(x) = sum_y w_y f_a(y) K(x,y)``

    as the value at ``x`` of a circular convolution.  ``K`` is either
    ``grad_x u`` (the default, returning a trailing Cartesian axis) or
    ``|grad_x u|^2``.  The left-coordinate dependence is retained exactly;
    later optimisations may factor it into Boys--Handy envelope channels.
    """
    mesh = _validate_uniform_grid(grid_points, weights, mesh)
    grid = jnp.asarray(grid_points)
    weights = jnp.asarray(weights)
    right = jnp.asarray(right_functions)
    if right.ndim == 1:
        right = right[None, :]
    if right.ndim != 2 or right.shape[1] != grid.shape[0]:
        raise ValueError(
            "right_functions must have shape (n_rhs, n_grid) or (n_grid,)"
        )

    axes = (-3, -2, -1)
    rhs_hat = jnp.fft.fftn((right * weights[None, :]).reshape((-1,) + mesh), axes=axes)
    values = []
    for left_flat in range(grid.shape[0]):
        gradients_by_y = jastrow_factor.grad_r_batch(
            grid[left_flat : left_flat + 1], grid, jastrow_params
        )[0]
        if squared_gradient:
            kernel_by_y = jnp.sum(gradients_by_y * gradients_by_y, axis=-1)
        else:
            kernel_by_y = gradients_by_y

        source = jnp.asarray(_source_indices(left_flat, mesh))
        kernel_by_displacement = kernel_by_y[source]
        left_index = np.unravel_index(left_flat, mesh)

        if squared_gradient:
            kernel_hat = jnp.fft.fftn(
                kernel_by_displacement.reshape(mesh), axes=axes
            )
            convolved = jnp.fft.ifftn(kernel_hat[None, ...] * rhs_hat, axes=axes)
            values.append(convolved[(slice(None),) + left_index])
        else:
            kernel_hat = jnp.fft.fftn(
                jnp.moveaxis(kernel_by_displacement.reshape(mesh + (3,)), -1, 0),
                axes=axes,
            )
            convolved = jnp.fft.ifftn(
                kernel_hat[:, None, ...] * rhs_hat[None, ...], axes=axes
            )
            values.append(convolved[(slice(None), slice(None)) + left_index].T)

    result = jnp.stack(values, axis=1)
    if not (jnp.iscomplexobj(right) or jnp.iscomplexobj(grid)):
        result = result.real
    return result


def calc_isdf_kernels_fft(
    xi_phi,
    xi_grad,
    weights,
    grid_points,
    mesh,
    jastrow_factor,
    jastrow_params,
):
    """Build ISDF ``U1`` and ``U3`` kernels with the FFT pair backend.

    The definitions match :func:`pytc.kmat.calc_K1_kernel` and
    :func:`pytc.kmat.calc_K3_kernel`.  Keeping this function independent of an
    ``ISDFTC`` object makes shared-pivot construction and direct-oracle tests
    possible before the production storage policy is chosen.
    """
    xi_phi = jnp.asarray(xi_phi)
    xi_grad = jnp.asarray(xi_grad)
    weights = jnp.asarray(weights)
    if xi_phi.ndim != 2:
        raise ValueError("xi_phi must have shape (n_rank, n_grid)")
    if xi_grad.shape != xi_phi.shape + (3,):
        raise ValueError("xi_grad must have shape (n_rank, n_grid, 3)")

    gradient_potential = fft_pair_potential(
        grid_points,
        weights,
        mesh,
        jastrow_factor,
        jastrow_params,
        xi_phi,
    )
    squared_potential = fft_pair_potential(
        grid_points,
        weights,
        mesh,
        jastrow_factor,
        jastrow_params,
        xi_phi,
        squared_gradient=True,
    )
    u1 = jnp.einsum(
        "kgc,lgc,g->klc", xi_grad, gradient_potential, weights
    )
    u3 = jnp.einsum("kg,lg,g->kl", xi_phi, squared_potential, weights)
    return u1, u3


def _fft_v_block(obj, jastrow_params, rows: slice, cols: slice):
    phi_rows = obj.phi[rows]
    phi_cols = obj.phi[cols]
    right = jnp.einsum("rg,sg->rsg", phi_rows, phi_cols)
    flat = right.reshape((-1, obj.grid_points.shape[0]))
    potential = fft_pair_potential(
        obj.grid_points,
        obj.weights,
        obj.fft_mesh,
        obj.jastrow_factor,
        jastrow_params,
        flat,
    )
    return potential.reshape(
        (phi_rows.shape[0], phi_cols.shape[0], obj.grid_points.shape[0], 3)
    )


def _fft_k1(obj, jastrow_params, ranges):
    p, q, r, s = ranges
    phi_p = obj.phi[p]
    phi_q = obj.phi[q]
    phi_r = obj.phi[r]
    phi_s = obj.phi[s]
    grad_p = obj.grad_phi[p]
    right = jnp.einsum("rg,sg->rsg", phi_r, phi_s)
    potential = fft_pair_potential(
        obj.grid_points,
        obj.weights,
        obj.fft_mesh,
        obj.jastrow_factor,
        jastrow_params,
        right.reshape((-1, obj.grid_points.shape[0])),
    )
    left = jnp.einsum("pgd,qg->pqgd", grad_p, phi_q)
    value = jnp.einsum("pqgd,agd,g->pqa", left, potential, obj.weights)
    return value.reshape(
        (phi_p.shape[0], phi_q.shape[0], phi_r.shape[0], phi_s.shape[0])
    )


def _fft_k3(obj, jastrow_params, ranges):
    p, q, r, s = ranges
    phi_p = obj.phi[p]
    phi_q = obj.phi[q]
    phi_r = obj.phi[r]
    phi_s = obj.phi[s]
    right = jnp.einsum("rg,sg->rsg", phi_r, phi_s)
    potential = fft_pair_potential(
        obj.grid_points,
        obj.weights,
        obj.fft_mesh,
        obj.jastrow_factor,
        jastrow_params,
        right.reshape((-1, obj.grid_points.shape[0])),
        squared_gradient=True,
    )
    left = jnp.einsum("pg,qg->pqg", phi_p, phi_q)
    value = jnp.einsum("pqg,ag,g->pqa", left, potential, obj.weights)
    return value.reshape(
        (phi_p.shape[0], phi_q.shape[0], phi_r.shape[0], phi_s.shape[0])
    )


def _tc_block_fft(obj, jastrow_params, ranges):
    k1 = _fft_k1(obj, jastrow_params, ranges)
    if ranges[0] == ranges[1]:
        k2 = k1.transpose(1, 0, 2, 3)
    else:
        swapped = (ranges[1], ranges[0], ranges[2], ranges[3])
        k2 = _fft_k1(obj, jastrow_params, swapped).transpose(1, 0, 2, 3)
    return 0.5 * (k1 - k2 + _fft_k3(obj, jastrow_params, ranges))


def _get_2b_fft(obj, jastrow_params, block_str=None, ranges=None):
    if ranges is None and block_str is not None:
        ranges = obj._get_block_ranges(block_str)
    if ranges is None:
        ranges = (slice(None),) * 4

    result = _tc_block_fft(obj, jastrow_params, ranges)
    if ranges[0] == ranges[2] and ranges[1] == ranges[3]:
        result = result + result.transpose(2, 3, 0, 1)
    else:
        transpose_ranges = (ranges[2], ranges[3], ranges[0], ranges[1])
        result = result + _tc_block_fft(
            obj, jastrow_params, transpose_ranges
        ).transpose(2, 3, 0, 1)
    return -result


def _delta_u_fft(obj, jastrow_params, dm1=None, ranges=None):
    if dm1 is None:
        dm1 = obj._get_mf_dm()
    if ranges is None:
        ranges = (slice(None),) * 4
    p, q, r, s = ranges
    occupied = slice(0, obj.nocc) if obj.nocc is not None else slice(None)
    occupations = jnp.diagonal(dm1)[occupied]

    phi = obj.phi
    phi_occ = phi[occupied]
    phi_p, phi_q, phi_r, phi_s = phi[p], phi[q], phi[r], phi[s]

    block_cache = {}

    def v_block(rows, cols):
        key = (
            (rows.start, rows.stop, rows.step),
            (cols.start, cols.stop, cols.step),
        )
        if key not in block_cache:
            block_cache[key] = _fft_v_block(obj, jastrow_params, rows, cols)
        return block_cache[key]

    v_occ_occ = v_block(occupied, occupied)
    v_pq = v_block(p, q)
    v_rs = v_block(r, s)
    v_occ_p = v_block(occupied, p)
    v_occ_q = v_block(occupied, q)
    v_occ_r = v_block(occupied, r)
    v_occ_s = v_block(occupied, s)

    v_kk = jnp.einsum("iigd->igd", v_occ_occ)
    w_vec = 2 * jnp.einsum("i,igd->gd", occupations, v_kk)
    w_bar = 2 * jnp.einsum("i,ig->g", occupations, phi_occ * phi_occ)

    def pair_terms(phi_a, phi_b, v_ab, v_occ_a, v_occ_b):
        z_bar = jnp.einsum(
            "i,iagd,ibgd->abg", occupations, v_occ_a, v_occ_b
        )
        phi_ka = jnp.einsum("ig,ag->iag", phi_occ, phi_a)
        phi_kb = jnp.einsum("ig,bg->ibg", phi_occ, phi_b)
        g_ab = jnp.einsum(
            "i,iag,ibgd->abgd", occupations, phi_ka, v_occ_b
        ) + jnp.einsum(
            "i,ibg,iagd->abgd", occupations, phi_kb, v_occ_a
        )
        a_ab = jnp.einsum("gd,abgd->abg", w_vec, v_ab) - z_bar
        b_ab = 0.5 * w_bar[None, None, :, None] * v_ab - g_ab
        return a_ab, b_ab

    a_pq, b_pq = pair_terms(phi_p, phi_q, v_pq, v_occ_p, v_occ_q)
    a_rs, b_rs = pair_terms(phi_r, phi_s, v_rs, v_occ_r, v_occ_s)
    phi_pq = jnp.einsum("pg,qg->pqg", phi_p, phi_q)
    phi_rs = jnp.einsum("rg,sg->rsg", phi_r, phi_s)
    weights = obj.weights
    result = jnp.einsum("pqg,rsg,g->pqrs", phi_pq, a_rs, weights)
    result += jnp.einsum("pqgd,rsgd,g->pqrs", v_pq, b_rs, weights)
    result += jnp.einsum("rsg,pqg,g->pqrs", phi_rs, a_pq, weights)
    result += jnp.einsum("rsgd,pqgd,g->pqrs", v_rs, b_pq, weights)
    return -result


def _get_3b_fock_fft(obj, jastrow_params, dm1):
    density = jnp.einsum("mg,ng,mn->g", obj.phi, obj.phi, dm1)
    potential = fft_pair_potential(
        obj.grid_points,
        obj.weights,
        obj.fft_mesh,
        obj.jastrow_factor,
        jastrow_params,
        density,
    )[0]
    local = jnp.sum(potential * potential, axis=-1)
    return jnp.einsum("pg,qg,g,g->pq", obj.phi, obj.phi, obj.weights, local)


@struct.dataclass
class FFTTC(TC):
    """Periodic TC object whose two- and three-body quadrature uses FFTs."""

    fft_mesh: tuple[int, int, int] = struct.field(pytree_node=False, default=None)

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        del batch_size
        return _get_2b_fft(self, jastrow_params, block_str, ranges)

    def get_3b_fock(self, jastrow_params, dm1):
        return _get_3b_fock_fft(self, jastrow_params, dm1)


@struct.dataclass
class FFTXTC(XTC):
    """Periodic xTC object whose TC and mean-field-reduction kernels use FFTs."""

    fft_mesh: tuple[int, int, int] = struct.field(pytree_node=False, default=None)

    def get_delta_U(self, jastrow_params, dm1=None, ranges=None, batch_size=1000):
        del batch_size
        return _delta_u_fft(self, jastrow_params, dm1, ranges)

    def get_2b(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        if dm1 is None:
            dm1 = self._get_mf_dm()
        tc_result = _get_2b_fft(self, jastrow_params, block_str, ranges)
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        return tc_result + self.get_delta_U(
            jastrow_params, dm1=dm1, ranges=ranges, batch_size=batch_size
        )

    def get_3b_fock(self, jastrow_params, dm1):
        return _get_3b_fock_fft(self, jastrow_params, dm1)


def _uniform_orbitals(mf, mo_coeff, mesh):
    cell = mf.cell
    if not cell.cart:
        raise ValueError("periodic TC requires a Cartesian basis")
    coefficients = _require_gamma_real(mf, mf.mo_coeff if mo_coeff is None else mo_coeff)
    mesh = _mesh_tuple(cell.mesh if mesh is None else mesh)
    coords = np.asarray(cell.gen_uniform_grids(mesh))
    weights = np.full(coords.shape[0], float(cell.vol) / coords.shape[0])
    ao = np.asarray(
        periodic_dft.numint.eval_ao_kpts(
            cell, coords, kpts=np.zeros((1, 3)), deriv=1
        )[0]
    )
    ao_values = ao[0].T.real
    ao_gradients = ao[1:4].transpose(2, 1, 0).real
    coefficients = jnp.asarray(coefficients)
    n_ao, n_orb = coefficients.shape
    phi = coefficients.T @ jnp.asarray(ao_values)
    grad_phi = (
        coefficients.T @ jnp.asarray(ao_gradients).reshape(n_ao, -1)
    ).reshape(n_orb, coords.shape[0], 3)
    return mesh, coords, weights, coefficients, phi, grad_phi


def create_tc_fft(mf, jastrow_factor, mo_coeff=None, mesh=None):
    """Create a real-Gamma periodic TC object on a uniform FFT mesh."""
    mesh, coords, weights, coefficients, phi, grad_phi = _uniform_orbitals(
        mf, mo_coeff, mesh
    )
    return FFTTC(
        grid_points=jnp.asarray(coords),
        weights=jnp.asarray(weights),
        phi=phi,
        grad_phi=grad_phi,
        n_orb=coefficients.shape[1],
        grid_lvl=-1,
        jastrow_factor=jastrow_factor,
        mo_coeff=coefficients,
        nocc=int(np.sum(mf.mo_occ > 0)),
        fft_mesh=mesh,
    )


def create_xtc_fft(mf, jastrow_factor, mo_coeff=None, mesh=None):
    """Create a real-Gamma periodic xTC object on a uniform FFT mesh."""
    tc = create_tc_fft(mf, jastrow_factor, mo_coeff=mo_coeff, mesh=mesh)
    return FFTXTC(
        grid_points=tc.grid_points,
        weights=tc.weights,
        phi=tc.phi,
        grad_phi=tc.grad_phi,
        n_orb=tc.n_orb,
        grid_lvl=tc.grid_lvl,
        jastrow_factor=tc.jastrow_factor,
        mo_coeff=tc.mo_coeff,
        nocc=tc.nocc,
        fft_mesh=tc.fft_mesh,
        mo_occ=jnp.asarray(mf.mo_occ),
        energy_nuc=float(mf.energy_nuc()),
    )
