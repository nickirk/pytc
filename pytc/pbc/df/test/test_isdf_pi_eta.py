"""Tests for pytc.pbc.df.isdf.build_pi_eta (task #21, design v2.1
sections 4/5)."""

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import build_pi_eta, build_pi_eta_staged
from pytc.pbc.df.kpts import canonicalize_kpts, pair_convolve


def _make_cell():
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag([2.0, 2.0, 2.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.build()
    return cell


def _tr_symmetric_fixture(rng, n_kpts, neg, shape):
    X = np.zeros((n_kpts,) + shape, dtype=np.complex128)
    done = set()
    for k in range(n_kpts):
        if k in done:
            continue
        nk = int(neg[k])
        if nk == k:
            X[k] = rng.normal(size=shape)
        else:
            re, im = rng.normal(size=shape), rng.normal(size=shape)
            X[k] = re + 1j * im
            X[nk] = re - 1j * im
            done.add(nk)
        done.add(k)
    return X


class TestBuildPiEta(unittest.TestCase):
    def test_pi_matches_direct_pair_convolve(self):
        cell = _make_cell()
        rng = np.random.default_rng(50)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao, n_g = 3, 5, 7
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_g, n_ao))

        Pi, eta = build_pi_eta(X, ao, mesh.phase, mesh.neg)
        # both Pi and eta are relabeled by neg relative to the raw
        # pair_convolve output (unified q<->-q convention fix, C2 item 2b).
        Pi_expected = pair_convolve(X, X, mesh.phase)[mesh.neg]
        eta_expected = pair_convolve(X, ao, mesh.phase)[mesh.neg]
        np.testing.assert_allclose(Pi, Pi_expected, atol=1e-12)
        np.testing.assert_allclose(eta, eta_expected, atol=1e-12)

    def test_pi_is_hermitian_per_q(self):
        cell = _make_cell()
        rng = np.random.default_rng(51)
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao = 4, 6
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (5, n_ao))
        Pi, _ = build_pi_eta(X, ao, mesh.phase, mesh.neg)
        for q in range(mesh.n_kpts):
            np.testing.assert_allclose(Pi[q], Pi[q].conj().T, atol=1e-10, err_msg=f"q={q}")

    def test_pi_neg_q_conj_closure(self):
        cell = _make_cell()
        rng = np.random.default_rng(52)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao = 3, 4
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (5, n_ao))
        Pi, eta = build_pi_eta(X, ao, mesh.phase, mesh.neg)
        np.testing.assert_allclose(Pi[mesh.neg], Pi.conj(), atol=1e-10)
        np.testing.assert_allclose(eta[mesh.neg], eta.conj(), atol=1e-10)

    def test_grid_block_accumulation_matches_single_block(self):
        cell = _make_cell()
        rng = np.random.default_rng(53)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        n_ip, n_ao, n_g = 3, 4, 9
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_ip, n_ao))
        ao = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (n_g, n_ao))

        Pi_single, eta_single = build_pi_eta(X, ao, mesh.phase, mesh.neg)
        blocks = [ao[:, :4, :], ao[:, 4:7, :], ao[:, 7:, :]]
        Pi_blocked, eta_blocked = build_pi_eta(X, blocks, mesh.phase, mesh.neg)

        np.testing.assert_allclose(Pi_blocked, Pi_single, atol=1e-12)
        np.testing.assert_allclose(eta_blocked, eta_single, atol=1e-12)

    def test_rejects_malformed_x_shape(self):
        cell = _make_cell()
        rng = np.random.default_rng(54)
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        X_bad = rng.normal(size=(mesh.n_kpts, 3)).astype(np.complex128)
        ao = rng.normal(size=(mesh.n_kpts, 5, 4)).astype(np.complex128)
        with self.assertRaises(ValueError):
            build_pi_eta(X_bad, ao, mesh.phase, mesh.neg)

    def test_rejects_empty_ao_blocks(self):
        cell = _make_cell()
        rng = np.random.default_rng(55)
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh = canonicalize_kpts(cell, kpts)
        X = _tr_symmetric_fixture(rng, mesh.n_kpts, mesh.neg, (3, 4))
        with self.assertRaises(ValueError):
            build_pi_eta(X, [], mesh.phase, mesh.neg)


if __name__ == "__main__":
    unittest.main()


class TestBuildUsesTheDeviceConvolve(unittest.TestCase):
    """The build uses the device convolve, and it agrees with the numpy reference.

    Replaces the old `convolve_device` flag tests. The flag is gone -- the numpy
    path was removed from the build (owner directive 2026-08-12) because mixing
    OpenBLAS with XLA cost 4.05x at 333, and a switch with one position is not a
    switch. But the numpy `pair_convolve` remains in kpts.py as the ORACLE, so
    what still needs testing is the invariant the flag used to carry:

      * the build reaches the device path, and
      * the device path still equals the numpy one.

    Asserting the second without the first would pass even if the build had
    quietly reverted to numpy, which is precisely the regression the old
    "flag reaches the staged builder" test existed to catch."""

    @staticmethod
    def _inputs(n_k=8, n_ip=60, n_ao=20, n_grid=240):
        rng = np.random.default_rng(0)

        def _c(*shape):
            return (rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(np.complex128)

        neg = np.arange(n_k)
        # Symmetrise so both paths clear their time-reversal gate.
        X = _c(n_k, n_ip, n_ao)
        ao = _c(n_k, n_grid, n_ao)
        X = np.stack([(X[k] + np.conj(X[neg[k]])) / 2 for k in range(n_k)])
        ao = np.stack([(ao[k] + np.conj(ao[neg[k]])) / 2 for k in range(n_k)])
        phase = np.exp(2j * np.pi * np.outer(np.arange(n_k), np.arange(n_k)) / n_k)
        return X, ao, phase / np.sqrt(n_k), neg

    def test_build_output_matches_the_numpy_oracle(self):
        from pytc.pbc.df.kpts import pair_convolve
        X, ao, phase, neg = self._inputs()
        built = build_pi_eta(X, [ao], phase, neg, imag_tol=1e30)
        want_pi = pair_convolve(X, X, phase, imag_tol=1e30)[neg]
        want_eta = pair_convolve(X, ao, phase, imag_tol=1e30)[neg]
        for name, got, want in (("Pi", built[0], want_pi),
                                ("eta", built[1], want_eta)):
            with self.subTest(name=name):
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_the_build_actually_calls_the_device_path(self):
        # The equality test above would pass even if the build had silently
        # reverted to numpy, since the two agree. This is the half that cannot:
        # it asserts WHICH function the build resolves to.
        from pytc.pbc.df import isdf as _isdf
        from pytc.pbc.df.kpts import pair_convolve_device
        self.assertIs(_isdf._pair_convolve(), pair_convolve_device)

    def test_no_convolve_selector_survives_in_the_build_api(self):
        # The numpy path is gone, so a keyword selecting it must be gone too --
        # a lingering no-op keyword is how a caller comes to believe it chose
        # something. Covers the staged builder, which previously imported
        # pair_convolve directly and could not be switched at all.
        import inspect
        for fn in (build_pi_eta, build_pi_eta_staged):
            with self.subTest(fn=fn.__name__):
                self.assertNotIn("convolve_device", inspect.signature(fn).parameters)
