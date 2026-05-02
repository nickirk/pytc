import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from .bh import BoysHandy, BHTerm


@struct.dataclass
class BoysHandyAnalytical(BoysHandy):
    """Work area for an analytical Boys-Handy implementation.

    This class is kept separate from ``BoysHandy`` so the original autodiff
    implementation remains available as a reference baseline for numerical
    comparisons while the analytical derivative path is developed.
    """
    padded_nuclei_by_type: jax.Array = struct.field(default=None)
    nuclei_mask_by_type: jax.Array = struct.field(default=None)

    @classmethod
    def create(cls, mol, terms_per_nucleus=None, epsilon=1e-8, name=None):
        inst = super().create(
            mol,
            terms_per_nucleus=terms_per_nucleus,
            epsilon=epsilon,
            name=name,
        )
        max_nuclei = max((group.shape[0] for group in inst.nuclei_by_type), default=0)
        padded_groups = []
        mask_groups = []
        for group in inst.nuclei_by_type:
            pad_n = max_nuclei - group.shape[0]
            padded_groups.append(jnp.pad(group, ((0, pad_n), (0, 0))))
            mask_groups.append(
                jnp.concatenate([
                    jnp.ones(group.shape[0], dtype=bool),
                    jnp.zeros(pad_n, dtype=bool),
                ])
            )
        if padded_groups:
            padded_nuclei = jnp.stack(padded_groups, axis=0)
            nuclei_mask = jnp.stack(mask_groups, axis=0)
        else:
            padded_nuclei = jnp.zeros((0, 0, 3))
            nuclei_mask = jnp.zeros((0, 0), dtype=bool)
        return inst.replace(
            padded_nuclei_by_type=padded_nuclei,
            nuclei_mask_by_type=nuclei_mask,
        )

    def _scaled_r_and_derivs(self, r_ref, r_target, scale):
        """Return scaled distance plus gradient/laplacian wrt ``r_ref``."""
        diff = r_ref - r_target
        dist = self._safe_norm(diff)
        denom = 1.0 + scale * dist
        f = scale * dist / denom
        
        f_d1 = scale / (denom**2)
        f_d2 = -2.0 * (scale**2) / (denom**3)
        
        unit = diff / dist[..., None]
        grad = f_d1[..., None] * unit
        grad_sq = jnp.sum(grad**2, axis=-1)
        lap = f_d2 + 2.0 * f_d1 / dist
        return f, grad, grad_sq, lap

    def _compute_forward(self, r1, r2, params):
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']
        # Divide by natom so that sum over atoms gives exactly 0.5 (matches
        # bh.py:_compute_forward and bha.py:get_log_grads_r1; commit 7cc4f56
        # missed this site).
        c = jnp.where(self._cusp_mask, 0.5 / self.natom, c_raw)

        exponents = jnp.arange(self.max_degree + 1)

        def get_powers(x):
            return jnp.power(x[..., None], exponents)
        nuclei = self.padded_nuclei_by_type
        nuclei_mask = self.nuclei_mask_by_type

        r1I = self._scaled_r_en(r1[None, None, :], nuclei, b[:, None])
        r2I = self._scaled_r_en(r2[None, None, :], nuclei, b[:, None])
        r12 = self._scaled_r_ee(r1[None, :], r2[None, :], d)

        p_r1I = get_powers(r1I)
        p_r2I = get_powers(r2I)
        p_r12 = get_powers(r12)

        v_r1I_m = jnp.take_along_axis(p_r1I, self._term_m[:, None, :], axis=2)
        v_r2I_n = jnp.take_along_axis(p_r2I, self._term_n[:, None, :], axis=2)
        v_r2I_m = jnp.take_along_axis(p_r2I, self._term_m[:, None, :], axis=2)
        v_r1I_n = jnp.take_along_axis(p_r1I, self._term_n[:, None, :], axis=2)
        v_r12_o = jnp.take_along_axis(p_r12, self._term_o, axis=1)[:, None, :]

        non_cusp_term = (v_r1I_m * v_r2I_n + v_r2I_m * v_r1I_n) * v_r12_o
        cusp_term = (2.0 / d)[:, None, None] * v_r12_o
        term_vals = jnp.where(self._cusp_mask[:, None, :], cusp_term, non_cusp_term)

        weights = self._delta_factor[:, None, :] * c[:, None, :]
        term_vals = jnp.where(nuclei_mask[:, :, None], term_vals, 0.0)
        return jnp.sum(weights * term_vals)

    def _power_tables(self, scalar, grad_scalar, grad_scalar_sq, lap_scalar):
        """Return power, gradient, and laplacian tables up to ``max_degree``."""
        exponents = jnp.arange(self.max_degree + 1, dtype=scalar.dtype)
        scalar_expanded = scalar[..., None]
        powers = jnp.power(scalar_expanded, exponents)

        grad_coeff = jnp.where(exponents > 0, exponents * jnp.power(scalar_expanded, exponents - 1), 0.0)
        
        lap_coeff_1 = jnp.where(
            exponents > 1,
            exponents * (exponents - 1) * jnp.power(scalar_expanded, exponents - 2),
            0.0,
        )
        lap_coeff_2 = grad_coeff

        grad_powers = grad_coeff[..., None] * grad_scalar[..., None, :]
        lap_powers = lap_coeff_1 * grad_scalar_sq[..., None] + lap_coeff_2 * lap_scalar[..., None]
        return powers, grad_powers, lap_powers

    def get_log_grads_r1(self, r1, r2, params):
        """Compute BH gradient and laplacian analytically wrt ``r1``."""
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']
        # Divide by natom so that sum over atoms gives exactly 0.5
        c = jnp.where(self._cusp_mask, 0.5 / self.natom, c_raw)
        nuclei = self.padded_nuclei_by_type
        nuclei_mask = self.nuclei_mask_by_type
        degree_exponents = jnp.arange(self.max_degree + 1, dtype=r1.dtype)

        x, grad_x, grad_x_sq, lap_x = self._scaled_r_and_derivs(
            r1[None, None, :], nuclei, b[:, None]
        )
        y = self._scaled_r_en(r2[None, None, :], nuclei, b[:, None])
        z, grad_z, grad_z_sq, lap_z = self._scaled_r_and_derivs(
            r1[None, :], r2[None, :], d
        )

        y_pows = jnp.power(y[..., None], degree_exponents)
        x_pow, grad_x_pow, lap_x_pow = self._power_tables(x, grad_x, grad_x_sq, lap_x)
        z_pow, grad_z_pow, lap_z_pow = self._power_tables(z, grad_z, grad_z_sq, lap_z)

        xm = jnp.take_along_axis(x_pow, self._term_m[:, None, :], axis=2)
        grad_xm = jnp.take_along_axis(
            grad_x_pow, self._term_m[:, None, :, None], axis=2
        )
        lap_xm = jnp.take_along_axis(lap_x_pow, self._term_m[:, None, :], axis=2)

        xn = jnp.take_along_axis(x_pow, self._term_n[:, None, :], axis=2)
        grad_xn = jnp.take_along_axis(
            grad_x_pow, self._term_n[:, None, :, None], axis=2
        )
        lap_xn = jnp.take_along_axis(lap_x_pow, self._term_n[:, None, :], axis=2)

        yn = jnp.take_along_axis(y_pows, self._term_n[:, None, :], axis=2)
        ym = jnp.take_along_axis(y_pows, self._term_m[:, None, :], axis=2)

        zo = jnp.take_along_axis(z_pow, self._term_o, axis=1)[:, None, :]
        grad_zo = jnp.take_along_axis(
            grad_z_pow, self._term_o[:, :, None], axis=1
        )[:, None, :, :]
        lap_zo = jnp.take_along_axis(lap_z_pow, self._term_o, axis=1)[:, None, :]

        part1 = xm * yn
        grad_part1 = grad_xm * yn[..., None]
        lap_part1 = lap_xm * yn

        part2 = xn * ym
        grad_part2 = grad_xn * ym[..., None]
        lap_part2 = lap_xn * ym

        base = part1 + part2
        grad_base = grad_part1 + grad_part2
        lap_base = lap_part1 + lap_part2

        non_cusp_grad = grad_base * zo[..., None] + base[..., None] * grad_zo
        non_cusp_lap = (
            lap_base * zo
            + base * lap_zo
            + 2.0 * jnp.sum(grad_base * grad_zo, axis=-1)
        )

        cusp_grad = (2.0 / d)[:, None, None, None] * grad_zo
        cusp_lap = (2.0 / d)[:, None, None] * lap_zo

        term_grad = jnp.where(self._cusp_mask[:, None, :, None], cusp_grad, non_cusp_grad)
        term_lap = jnp.where(self._cusp_mask[:, None, :], cusp_lap, non_cusp_lap)

        weights = self._delta_factor[:, None, :] * c[:, None, :]
        term_grad = jnp.where(nuclei_mask[:, :, None, None], term_grad, 0.0)
        term_lap = jnp.where(nuclei_mask[:, :, None], term_lap, 0.0)

        total_grad = jnp.sum(weights[..., None] * term_grad, axis=(0, 1, 2))
        total_lap = jnp.sum(weights * term_lap)
        return total_grad, total_lap

    def get_log_grads_r2(self, r1, r2, params):
        return self.get_log_grads_r1(r2, r1, params)

    def grad_r(self, r1, r2, params):
        return self.get_log_grads_r1(r1, r2, params)[0]

    def laplacian_r(self, r1, r2, params):
        return self.get_log_grads_r1(r1, r2, params)[1]
