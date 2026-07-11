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
    def create(cls, mol, terms_per_nucleus=None, epsilon=1e-16, name=None):
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
        """Return scaled distance plus gradient/laplacian wrt ``r_ref``.

        ``lap`` includes a ``2*f_d1/dist`` term -- ``dist`` (from
        ``_safe_norm``) must not be floored above the true near-coalescence
        scale this is meant to capture, or the laplacian incorrectly
        plateaus instead of diverging as r->0 (the correct cusp behavior;
        confirmed against BoysHandy's autodiff reference). This is why
        ``epsilon`` must match BoysHandy's convention (see ``create``).
        """
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

    def get_pair_grid_grad_lap(self, elec_coords, params):
        """Whole-electron-set analytic grad/laplacian for ALL (i,j) pairs at once.

        Replaces the per-pair call pattern (``get_log_grads_r1`` invoked once
        per (i,j) pair via a vmap grid in ``compute_jastrow_terms``), which
        redundantly recomputes electron i's atom-distance table on every one
        of the N pairs i appears in -- O(N^2*M) work. Here, per-(electron,
        atom) power/gradient/laplacian tables are built ONCE (O(N*M)), and
        the pair grid is assembled by scanning over atoms and accumulating
        into an (N,N,3)/(N,N) carry, so the O(n_terms) factor stays a
        per-step transient rather than a permanent O(N^2*natom*n_terms)
        tensor (task #5 PR-B, #pro-pytc-efficiency-refactor).

        Args:
            elec_coords: (N, 3) all electron positions.
            params: Jastrow parameters.

        Returns:
            (grad_pair, lap_pair): grad_pair has shape (N, N, 3) and
            lap_pair has shape (N, N), where [i, j] holds grad_1/lap_1 of
            u(r_i, r_j) wrt r_i -- the same quantities and layout that
            ``compute_jastrow_terms``'s vmap grid produces (BEFORE the
            diagonal mask and the sum-over-j reduction it applies next).

        Note: at very small N (measured on A100: N~20, e.g. H20), the
        fixed per-atom scan overhead here can be marginally slower on the
        Jacobian phase than the base class's plain per-pair vmap grid
        (H20: 0.044s pairwise vs 0.18s here -- both trivial in absolute
        terms). The crossover is between N=20 and N=40; users targeting
        small systems who care about that margin can construct generic
        ``BoysHandy`` instead. No implicit switching between the two --
        class choice is the only dispatch, per pytc's explicit-choice
        design (task #5, #proj-pytc-efficiency-refactor).
        """
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']
        c = jnp.where(self._cusp_mask, 0.5 / self.natom, c_raw)

        atom_type_map = self.atom_type_map  # (natom,)
        nuclear_pos = self.nuclear_pos       # (natom, 3)
        b_atom = b[atom_type_map]            # (natom,)
        d_atom = d[atom_type_map]            # (natom,)
        weight_atom = (self._delta_factor * c)[atom_type_map]   # (natom, n_terms)
        cusp_mask_atom = self._cusp_mask[atom_type_map]          # (natom, n_terms)
        term_m_atom = self._term_m[atom_type_map]                # (natom, n_terms)
        term_n_atom = self._term_n[atom_type_map]                # (natom, n_terms)

        # --- Per-(electron, atom) tables, built once: O(N * natom) ---
        r_elec = elec_coords[:, None, :]          # (N, 1, 3)
        r_nuc = nuclear_pos[None, :, :]            # (1, natom, 3)
        diff = r_elec - r_nuc                      # (N, natom, 3)
        dist = self._safe_norm(diff)                # (N, natom)
        denom = 1.0 + b_atom[None, :] * dist         # (N, natom)
        x = b_atom[None, :] * dist / denom
        f_d1 = b_atom[None, :] / (denom ** 2)
        f_d2 = -2.0 * (b_atom[None, :] ** 2) / (denom ** 3)
        unit = diff / dist[..., None]
        grad_x = f_d1[..., None] * unit             # (N, natom, 3)
        grad_x_sq = jnp.sum(grad_x ** 2, axis=-1)    # (N, natom)
        lap_x = f_d2 + 2.0 * f_d1 / dist             # (N, natom)

        x_pow, grad_x_pow, lap_x_pow = self._power_tables(x, grad_x, grad_x_sq, lap_x)
        # x_pow: (N, natom, deg+1); grad_x_pow: (N, natom, deg+1, 3); lap_x_pow: (N, natom, deg+1)

        xm_val = jnp.take_along_axis(x_pow, term_m_atom[None, :, :], axis=2)          # (N, natom, n_terms)
        grad_xm = jnp.take_along_axis(grad_x_pow, term_m_atom[None, :, :, None], axis=2)  # (N, natom, n_terms, 3)
        lap_xm = jnp.take_along_axis(lap_x_pow, term_m_atom[None, :, :], axis=2)      # (N, natom, n_terms)

        xn_val = jnp.take_along_axis(x_pow, term_n_atom[None, :, :], axis=2)
        grad_xn = jnp.take_along_axis(grad_x_pow, term_n_atom[None, :, :, None], axis=2)
        lap_xn = jnp.take_along_axis(lap_x_pow, term_n_atom[None, :, :], axis=2)

        # --- Per-(pair, TYPE) e-e tables, built once: O(N^2 * n_types) ---
        # r12 only depends on the atom's TYPE (via d), not its individual
        # position, so this is shared across all atoms of the same type
        # instead of being recomputed per atom.
        r_i = elec_coords[:, None, None, :]   # (N, 1, 1, 3)
        r_j = elec_coords[None, :, None, :]   # (1, N, 1, 3)
        d_b = d[None, None, :]                # (1, 1, n_types)
        z, grad_z, grad_z_sq, lap_z = self._scaled_r_and_derivs(r_i, r_j, d_b)
        # z: (N, N, n_types); grad_z: (N, N, n_types, 3); lap_z: (N, N, n_types)

        z_pow, grad_z_pow, lap_z_pow = self._power_tables(z, grad_z, grad_z_sq, lap_z)
        # z_pow: (N, N, n_types, deg+1); grad_z_pow: (..., 3); lap_z_pow: (N, N, n_types, deg+1)

        term_o_b = self._term_o[None, None, :, :]  # (1, 1, n_types, n_terms)
        zo_type = jnp.take_along_axis(z_pow, term_o_b, axis=3)                                   # (N, N, n_types, n_terms)
        grad_zo_type = jnp.take_along_axis(grad_z_pow, term_o_b[..., None], axis=3)              # (N, N, n_types, n_terms, 3)
        lap_zo_type = jnp.take_along_axis(lap_z_pow, term_o_b, axis=3)                            # (N, N, n_types, n_terms)

        # --- Scan over atoms, accumulating the (N, N, 3)/(N, N) pair grid ---
        # Keeps peak transient memory at O(N^2 * n_terms) per step instead
        # of O(N^2 * natom * n_terms) for the whole grid at once.
        n_elec = elec_coords.shape[0]

        def scan_body(carry, xs_a):
            grad_carry, lap_carry = carry
            (xm_a, grad_xm_a, lap_xm_a, xn_a, grad_xn_a, lap_xn_a,
             atom_type_a, weight_a, cusp_mask_a, d_a) = xs_a
            # xm_a etc: (N, n_terms) for this atom, electron axis N.

            zo = jax.lax.dynamic_index_in_dim(zo_type, atom_type_a, axis=2, keepdims=False)       # (N, N, n_terms)
            grad_zo = jax.lax.dynamic_index_in_dim(grad_zo_type, atom_type_a, axis=2, keepdims=False)  # (N, N, n_terms, 3)
            lap_zo = jax.lax.dynamic_index_in_dim(lap_zo_type, atom_type_a, axis=2, keepdims=False)    # (N, N, n_terms)

            base = xm_a[:, None, :] * xn_a[None, :, :] + xn_a[:, None, :] * xm_a[None, :, :]       # (N, N, n_terms)
            grad_base = (grad_xm_a[:, None, :, :] * xn_a[None, :, :, None]
                         + grad_xn_a[:, None, :, :] * xm_a[None, :, :, None])                      # (N, N, n_terms, 3)
            lap_base = lap_xm_a[:, None, :] * xn_a[None, :, :] + lap_xn_a[:, None, :] * xm_a[None, :, :]  # (N, N, n_terms)

            non_cusp_grad = grad_base * zo[..., None] + base[..., None] * grad_zo
            non_cusp_lap = lap_base * zo + base * lap_zo + 2.0 * jnp.sum(grad_base * grad_zo, axis=-1)

            cusp_grad = (2.0 / d_a) * grad_zo
            cusp_lap = (2.0 / d_a) * lap_zo

            cusp_mask_b = cusp_mask_a[None, None, :]
            term_grad = jnp.where(cusp_mask_b[..., None], cusp_grad, non_cusp_grad)   # (N, N, n_terms, 3)
            term_lap = jnp.where(cusp_mask_b, cusp_lap, non_cusp_lap)                  # (N, N, n_terms)

            atom_grad = jnp.sum(weight_a[None, None, :, None] * term_grad, axis=2)     # (N, N, 3)
            atom_lap = jnp.sum(weight_a[None, None, :] * term_lap, axis=2)             # (N, N)

            return (grad_carry + atom_grad, lap_carry + atom_lap), None

        init_carry = (
            jnp.zeros((n_elec, n_elec, 3), dtype=elec_coords.dtype),
            jnp.zeros((n_elec, n_elec), dtype=elec_coords.dtype),
        )
        xs = (
            jnp.moveaxis(xm_val, 1, 0), jnp.moveaxis(grad_xm, 1, 0), jnp.moveaxis(lap_xm, 1, 0),
            jnp.moveaxis(xn_val, 1, 0), jnp.moveaxis(grad_xn, 1, 0), jnp.moveaxis(lap_xn, 1, 0),
            atom_type_map, weight_atom, cusp_mask_atom, d_atom,
        )
        # jax.checkpoint (rematerialization): without this, reverse-mode AD
        # through lax.scan stores each step's forward intermediates (here,
        # O(N^2*n_terms) per atom) for ALL natom steps to compute the
        # backward pass -- reintroducing the O(N^2*natom*n_terms) memory
        # blowup the scan's O(N^2) carry was designed to avoid, just moved
        # from the forward pass to the backward pass. checkpoint trades
        # that storage for recomputing each step's forward pass during the
        # backward pass instead.
        (grad_pair, lap_pair), _ = jax.lax.scan(jax.checkpoint(scan_body), init_carry, xs)
        return grad_pair, lap_pair
