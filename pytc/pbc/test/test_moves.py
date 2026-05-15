import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.pbc.vmc.moves import (
    _all_electron_move,
    _one_electron_move,
    _compute_green_function,
)
from pytc.pbc.ansatz import create_slater_det
from pytc.pbc.utils import wrap, mic_displacement
from pytc.vmc.walker import initialize_walker_state
from pytc.ansatz.det import eval_det_value


class _ConstAnsatz:
    """Trivial ansatz returning constant ψ. Sufficient to exercise the move
    proposal/wrap path without needing a real PBC Slater determinant."""

    n_alpha = 1
    n_electrons = 2

    def __call__(self, walker, params):
        n_walkers = walker.positions.shape[0]
        psi_sign = jnp.ones((n_walkers,))
        log_psi = jnp.zeros((n_walkers,))
        walker_updated = walker.replace(
            log_psi=log_psi,
            psi_sign=psi_sign,
            log_jastrow=jnp.zeros((n_walkers,)),
        )
        return (psi_sign, log_psi), walker_updated


def _make_walker(n_walkers, lattice, key):
    ansatz = _ConstAnsatz()
    L = jnp.diag(lattice)
    positions = random.uniform(key, (n_walkers, 2, 3)) * L
    return initialize_walker_state(ansatz, positions), ansatz


class TestAllElectronMove(unittest.TestCase):
    def test_proposals_inside_cell(self):
        lattice = jnp.diag(jnp.array([3.0, 4.0, 5.0]))
        walker, ansatz = _make_walker(128, lattice, random.PRNGKey(0))
        # Large step size to make sure proposals routinely leave the cell
        # before wrapping.
        _, _, _, proposals = _all_electron_move(
            ansatz, walker, step_size=2.5, key=random.PRNGKey(1),
            params=None, lattice=lattice,
        )
        L = jnp.diag(lattice)
        self.assertTrue(bool(jnp.all(proposals.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(proposals.positions < L)))

    def test_zero_step_leaves_positions_invariant(self):
        lattice = jnp.diag(jnp.array([3.0, 4.0, 5.0]))
        walker, ansatz = _make_walker(8, lattice, random.PRNGKey(2))
        _, _, _, proposals = _all_electron_move(
            ansatz, walker, step_size=0.0, key=random.PRNGKey(3),
            params=None, lattice=lattice,
        )
        # Step size 0 → proposals == walker.positions (after the no-op wrap).
        np.testing.assert_allclose(
            proposals.positions, walker.positions, atol=1e-12
        )

    def test_move_mask_all_true(self):
        lattice = jnp.diag(jnp.array([3.0, 3.0, 3.0]))
        walker, ansatz = _make_walker(4, lattice, random.PRNGKey(4))
        _, _, _, proposals = _all_electron_move(
            ansatz, walker, step_size=0.5, key=random.PRNGKey(5),
            params=None, lattice=lattice,
        )
        self.assertTrue(bool(jnp.all(proposals.move_mask)))


class TestGreenFunctionPBC(unittest.TestCase):
    def test_matches_molecular_for_small_displacement(self):
        """When the source-target displacement is well inside the cell, the
        PBC Green's function should reduce to the molecular one."""
        from pytc.vmc.moves import _compute_green_function as molecular_g

        lattice = jnp.diag(jnp.array([100.0, 100.0, 100.0]))  # huge cell
        rng = np.random.default_rng(0)
        r_source = jnp.asarray(rng.uniform(0, 1, size=(8, 4, 3)))
        r_target = r_source + jnp.asarray(rng.normal(scale=0.1, size=(8, 4, 3)))
        qf = jnp.asarray(rng.normal(size=(8, 4, 3)) * 0.05)
        tau = 0.05

        g_pbc = _compute_green_function(r_target, r_source, qf, tau, lattice)
        g_mol = molecular_g(r_target, r_source, qf, tau)
        np.testing.assert_allclose(g_pbc, g_mol, rtol=1e-10)

    def test_periodic_in_target(self):
        """G should be invariant under adding a lattice vector to r_target."""
        lattice = jnp.diag(jnp.array([4.0, 5.0, 6.0]))
        rng = np.random.default_rng(1)
        r_source = jnp.asarray(rng.uniform(0, 4, size=(3, 2, 3)))
        r_target = jnp.asarray(rng.uniform(0, 4, size=(3, 2, 3)))
        qf = jnp.zeros_like(r_source)
        tau = 0.1

        g1 = _compute_green_function(r_target, r_source, qf, tau, lattice)

        # Translate target by lattice vector a1
        a1 = lattice[0]
        g2 = _compute_green_function(r_target + a1, r_source, qf, tau, lattice)
        np.testing.assert_allclose(g1, g2, rtol=1e-10)

    def test_uses_minimum_image(self):
        """A target that crosses the cell boundary should still register as
        a short hop, not a long one."""
        lattice = jnp.diag(jnp.array([4.0, 4.0, 4.0]))
        # Source near +x face, target near -x face: raw diff is ~3.9, MIC ~0.1
        r_source = jnp.array([[[3.95, 2.0, 2.0]]])
        r_target = jnp.array([[[0.05, 2.0, 2.0]]])
        qf = jnp.zeros_like(r_source)
        tau = 1.0

        g_pbc = _compute_green_function(r_target, r_source, qf, tau, lattice)
        # MIC distance = 0.1 → exponent = -0.01/2 → G ≈ exp(-0.005)
        expected = jnp.exp(-(0.1 ** 2) / (2.0 * tau))
        np.testing.assert_allclose(g_pbc[0], expected, rtol=1e-10)

    def test_jit_compatible(self):
        lattice = jnp.diag(jnp.array([4.0, 5.0, 6.0]))
        rng = np.random.default_rng(2)
        r_source = jnp.asarray(rng.uniform(0, 4, size=(2, 2, 3)))
        r_target = jnp.asarray(rng.uniform(0, 4, size=(2, 2, 3)))
        qf = jnp.zeros_like(r_source)
        f_jit = jax.jit(_compute_green_function, static_argnames=())
        g = f_jit(r_target, r_source, qf, 0.1, lattice)
        g_ref = _compute_green_function(r_target, r_source, qf, 0.1, lattice)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10)


def _h2_pbc_det(L=6.0):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    mf = pbcscf.RHF(cell)
    mf.exxdiv = None
    mf.kernel()
    det = create_slater_det(cell, mo_coeff=mf.mo_coeff)
    return cell, det


class TestOneElectronMove(unittest.TestCase):
    def setUp(self):
        self.cell, self.det = _h2_pbc_det(L=6.0)
        self.lattice = jnp.asarray(self.cell.lattice_vectors())
        L = jnp.diag(self.lattice)
        positions = random.uniform(random.PRNGKey(0), (16, 2, 3)) * L
        walker = initialize_walker_state(self.det, positions)
        # Prime the cache with eval_det_value via vmap
        _, walker = jax.vmap(lambda w: eval_det_value(self.det, w))(walker)
        self.walker = walker

    def test_proposals_inside_cell(self):
        _, _, _, proposals = _one_electron_move(
            self.det, self.walker, step_size=1.5, key=random.PRNGKey(1),
            params=None, lattice=self.lattice,
        )
        L = jnp.diag(self.lattice)
        self.assertTrue(bool(jnp.all(proposals.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(proposals.positions < L)))

    def test_exactly_one_electron_moves_per_walker(self):
        _, _, _, proposals = _one_electron_move(
            self.det, self.walker, step_size=0.3, key=random.PRNGKey(2),
            params=None, lattice=self.lattice,
        )
        # move_mask: exactly one True per walker
        counts = jnp.sum(proposals.move_mask.astype(jnp.int32), axis=-1)
        np.testing.assert_array_equal(np.asarray(counts), np.ones(16, dtype=np.int32))

    def test_unmoved_electrons_unchanged(self):
        _, _, _, proposals = _one_electron_move(
            self.det, self.walker, step_size=0.3, key=random.PRNGKey(3),
            params=None, lattice=self.lattice,
        )
        # Where move_mask is False, position must equal walker.positions
        unmoved = ~proposals.move_mask
        old = jnp.where(unmoved[:, :, None], self.walker.positions, 0.0)
        new = jnp.where(unmoved[:, :, None], proposals.positions, 0.0)
        np.testing.assert_allclose(new, old, atol=1e-12)

    def test_rank1_psi_matches_full_recompute(self):
        """The cached log_psi on proposals must match an independent full
        determinant evaluation at proposals.positions."""
        _, new_psi_values, _, proposals = _one_electron_move(
            self.det, self.walker, step_size=0.3, key=random.PRNGKey(4),
            params=None, lattice=self.lattice,
        )
        new_psi_sign, new_psi_logabs = new_psi_values

        # Independent full eval at proposed positions
        fresh = initialize_walker_state(self.det, proposals.positions)
        _, fresh = jax.vmap(lambda w: eval_det_value(self.det, w))(fresh)

        np.testing.assert_allclose(new_psi_logabs, fresh.log_psi, atol=1e-10)
        np.testing.assert_allclose(new_psi_sign, fresh.psi_sign, atol=1e-10)

    def test_psi_invariant_under_wrap(self):
        """A move whose proposed position would land outside the cell must
        give the same log|ψ| as one wrapped back into the cell, because the
        PBC orbital is periodic."""
        # Use a step size much larger than the cell so most proposals wrap.
        _, new_psi_values_a, _, _ = _one_electron_move(
            self.det, self.walker, step_size=3.0, key=random.PRNGKey(5),
            params=None, lattice=self.lattice,
        )
        # Run the same move on a walker whose positions are shifted by a
        # whole lattice vector — the answer must match.
        shifted_walker = self.walker.replace(
            positions=self.walker.positions + self.lattice[0]
        )
        _, _, _, proposals_shifted = _one_electron_move(
            self.det, shifted_walker, step_size=3.0, key=random.PRNGKey(5),
            params=None, lattice=self.lattice,
        )
        # Proposals should be identical after wrapping (the random proposal
        # is the same; only the wrapped result is stored).
        np.testing.assert_allclose(
            proposals_shifted.log_psi, new_psi_values_a[1], atol=1e-10
        )


if __name__ == '__main__':
    unittest.main()
