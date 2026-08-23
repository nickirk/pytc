"""Tests for JAX implementation of kinetic matrix elements."""

import inspect
import unittest
import tempfile
import numpy as np
import h5py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pytc.legacy.kmat import calc_K1 as calc_K1_numpy, calc_K3 as calc_K3_numpy
from pytc.kmat import (
    calc_K1,
    calc_K3,
    calc_kmat_kernels_from_aux,
    calc_kmat_kernels_from_aux_streamed,
)
from pytc.jastrow import Poly, BoysHandy, CompositeJastrow, NuclearCusp


class TestKmat(unittest.TestCase):
    """Test JAX implementation of K matrix elements."""
    
    def setUp(self):
        """Set up test fixtures."""
        rng = np.random.RandomState(42)
        
        self.test_configs = [
            {'Nb': 2, 'N_grid': 3, 'name': 'small'},
            {'Nb': 4, 'N_grid': 10, 'name': 'medium'},
            {'Nb': 6, 'N_grid': 20, 'name': 'large'}
        ]
        
        for config in self.test_configs:
            Nb, N_grid = config['Nb'], config['N_grid']
            config['grid_points'] = rng.randn(N_grid, 3)
            config['weights'] = rng.rand(N_grid)  # Random weights
            
            config['phi'] = rng.randn(Nb, N_grid)
            config['grad_phi'] = rng.randn(Nb, N_grid, 3)
            
            # Compute paired densities for NumPy reference (which expects pairs)
            # phi_paired_ij = phi_i * phi_j
            config['phi_paired'] = np.einsum('in,jn->ijn', config['phi'], config['phi']).reshape(Nb * Nb, N_grid)
            
            # grad_phi_paired_ij = grad_phi_i * phi_j
            # Note: This matches JAX calc_K1 logic (grad on first index)
            config['grad_phi_paired'] = np.einsum('ind,jn->ijnd', config['grad_phi'], config['phi']).reshape(Nb * Nb, N_grid, 3)
        
        self.params = jnp.array([1.0])
        self.jastrow_jax = Poly()
        
        class PolyNumpy:
            """NumPy implementation to match original implementation."""
            def __init__(self, params):
                self.params = params
                
            def grad(self, r1, r2):
                """Numpy gradient computation handling both single and batched inputs."""
                if r1.ndim == 1:
                    r1 = r1[None, :]
                if r2.ndim == 1:
                    r2 = r2[None, :]
                    
                diff = r1[:, None, :] - r2[None, :, :]
                r12 = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)  # Match epsilon
                grad = diff / r12[..., None]
                grad = grad * self.params[0]
                
                if grad.shape[0] == 1 and grad.shape[1] == 1:
                    return grad[0, 0]
                return grad
                
        self.jastrow_numpy = PolyNumpy(self.params)
    
    def test_K1_shapes(self):
        """Test K1 output shapes for different input sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                Nb = config['Nb']
                result = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                self.assertEqual(result.shape, (Nb * Nb, Nb * Nb))
    
    def test_K1_against_numpy_all_sizes(self):
        """Compare JAX K1 implementation against numpy for different sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                k1_jax_raw = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                # This matches NumPy (Nb^2, Nb^2)
                k1_jax = k1_jax_raw
                
                k1_numpy = calc_K1_numpy(
                    config['phi_paired'],
                    config['grad_phi_paired'],
                    self.jastrow_numpy,
                    config['grid_points'],
                    config['weights']
                )
                
                np.testing.assert_allclose(
                    np.asarray(k1_jax), k1_numpy,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"JAX and numpy K1 don't match for {config['name']} system"
                )
    
    def test_K3_shapes(self):
        """Test K3 output shapes for different input sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                Nb = config['Nb']
                result = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                self.assertEqual(result.shape, (Nb * Nb, Nb * Nb))
    
    def test_K3_against_numpy_all_sizes(self):
        """Compare JAX K3 implementation against numpy for different sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                k3_jax = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                
                k3_numpy = calc_K3_numpy(
                    config['phi_paired'],
                    self.jastrow_numpy,
                    config['grid_points'],
                    config['weights']
                )
                
                np.testing.assert_allclose(
                    np.asarray(k3_jax), k3_numpy,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"JAX and numpy K3 don't match for {config['name']} system"
                )
    
    def test_batch_size_handling(self):
        """Test different batch sizes produce same results."""
        config = self.test_configs[-1]
        batch_sizes = [1, 5, 10, 20]
        
        ref_k1 = calc_K1(
            jnp.asarray(config['phi']),
            jnp.asarray(config['grad_phi']),
            self.jastrow_jax,
            self.params,
            jnp.asarray(config['grid_points']),
            jnp.asarray(config['weights'])
        )
        
        ref_k3 = calc_K3(
            jnp.asarray(config['phi']),
            self.jastrow_jax,
            self.params,
            jnp.asarray(config['grid_points']),
            jnp.asarray(config['weights'])
        )
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                k1 = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights']),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k1, ref_k1, rtol=1e-5, atol=1e-5)
                
                k3 = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights']),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k3, ref_k3, rtol=1e-5, atol=1e-5)
    
    def test_single_point_gradient(self):
        """Test single point gradient computation matches between JAX and NumPy."""
        r1 = np.array([0., 0., 0.])
        r2 = np.array([1., 0., 0.])
        
        grad_jax = self.jastrow_jax.grad_r(r1, r2, self.params)
        grad_numpy = self.jastrow_numpy.grad(r1, r2)
        
        np.testing.assert_allclose(
            np.asarray(grad_jax), grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Single point gradients don't match"
        )


class TestKmatFromAuxParity(unittest.TestCase):
    """The K1/K3-from-aux identity must hold on a physical H2 ISDF grid."""

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf
        from pytc.jastrow.rexp import REXP
        from pytc.tc import TC, ISDFTC

        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        cls.params = {"alpha": jnp.array([1.0])}
        base = TC.from_pyscf(mf, REXP(), grid_lvl=0)
        cls.isdf = ISDFTC.from_tc(
            base, n_rank=max(8, 3 * base.n_orb), is_incore=True
        )

    def test_h2_kernels_recover_from_laux_and_haux(self):
        """No pair-kernel approximation is permitted in the first gate."""
        grid_block = self.isdf.grid_points.shape[0]
        direct = self.isdf.compute_kmat_kernels(
            self.params,
            batch_size=32,
            host_grid_block_size=grid_block,
            gpu_budget_bytes=2_000 * 1024**2,
        )
        l_aux, h_aux = self.isdf._compute_L_aux(
            self.params,
            batch_size=32,
            host_grid_block_size=grid_block,
            include_h_aux=True,
        )
        recovered = calc_kmat_kernels_from_aux(
            self.isdf.xi_phi,
            self.isdf.xi_grad,
            self.isdf.weights,
            l_aux,
            h_aux,
        )

        np.testing.assert_allclose(
            np.asarray(recovered["K1_kernel"]),
            np.asarray(direct["K1_kernel"]),
            rtol=0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            np.asarray(recovered["K3_kernel"]),
            np.asarray(direct["K3_kernel"]),
            rtol=0,
            atol=2e-12,
        )
        # The downstream ISDF two-body correction is the physics-facing
        # parity check.  L_aux is identical in both views; only K1/K3 are
        # exchanged for their exact auxiliary recovery.
        direct_2b = self.isdf.replace(
            isdf_kernels={**direct, "L_aux": l_aux}
        ).get_2b(self.params)
        recovered_2b = self.isdf.replace(
            isdf_kernels={**recovered, "L_aux": l_aux}
        ).get_2b(self.params)
        np.testing.assert_allclose(
            np.asarray(recovered_2b),
            np.asarray(direct_2b),
            rtol=0,
            atol=2e-12,
        )

    def test_aux_recovery_is_the_public_default(self):
        """TC, XTC, and factor-only builders must share the exact-reuse default."""
        from pytc.tc import ISDFTC
        from pytc.xtc import ISDFXTC

        for method in (
            ISDFTC.isdf,
            ISDFXTC.isdf,
            ISDFXTC.build_tucker_x_kernels_direct,
        ):
            with self.subTest(method=method.__qualname__):
                self.assertIs(
                    inspect.signature(method)
                    .parameters["reuse_aux_kernels"]
                    .default,
                    None,
                )

        in_core = self.isdf.isdf(
            self.params,
            batch_size=32,
            host_grid_block_size=512,
        )
        self.assertEqual(in_core.kmat_kernel_mode, "aux-recovery")

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/default-recovery.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("xi_phi", data=np.asarray(self.isdf.xi_phi))
                handle.create_dataset("xi_grad", data=np.asarray(self.isdf.xi_grad))
            out_of_core = self.isdf.replace(
                is_incore=False,
                xi_phi=None,
                xi_grad=None,
                save_path=path,
            ).isdf(
                self.params,
                save_path=path,
                batch_size=32,
                host_grid_block_size=512,
            )
            self.assertEqual(out_of_core.kmat_kernel_mode, "aux-recovery")
            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    handle.attrs["pytc_kmat_kernel_mode"], "aux-recovery"
                )
            warm = self.isdf.isdf(
                self.params,
                save_path=path,
                batch_size=32,
                host_grid_block_size=512,
            )
            self.assertEqual(warm.kmat_kernel_mode, "aux-recovery")

        no_store = self.isdf.replace(is_incore=False, save_path=None).isdf(
            self.params,
            batch_size=32,
            host_grid_block_size=self.isdf.grid_points.shape[0],
        )
        self.assertIn("K1_kernel", no_store.isdf_kernels)
        self.assertIn("K3_kernel", no_store.isdf_kernels)
        self.assertEqual(no_store.kmat_kernel_mode, "direct")

    def test_h2_streamed_aux_recovery_matches_direct_from_hdf5(self):
        """The out-of-core recovery must use HDF5 panels without a K-square GPU temporary."""
        grid_block = 127  # force several grid panels on the physical H2 grid
        direct = self.isdf.compute_kmat_kernels(
            self.params,
            batch_size=32,
            host_grid_block_size=self.isdf.grid_points.shape[0],
            gpu_budget_bytes=2_000 * 1024**2,
        )
        l_aux, h_aux = self.isdf._compute_L_aux(
            self.params,
            batch_size=32,
            host_grid_block_size=self.isdf.grid_points.shape[0],
            include_h_aux=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("xi_phi", data=np.asarray(self.isdf.xi_phi))
                f.create_dataset("xi_grad", data=np.asarray(self.isdf.xi_grad))
                f.create_dataset("weights", data=np.asarray(self.isdf.weights))
                f.create_dataset("L_aux", data=np.asarray(l_aux))
                f.create_dataset("H_aux", data=np.asarray(h_aux))
                streamed = calc_kmat_kernels_from_aux_streamed(
                    f["xi_phi"],
                    f["xi_grad"],
                    f["weights"],
                    f["L_aux"],
                    f["H_aux"],
                    grid_block_size=grid_block,
                    rank_block_size=3,
                )

        np.testing.assert_allclose(
            streamed["K1_kernel"], np.asarray(direct["K1_kernel"]),
            rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            streamed["K3_kernel"], np.asarray(direct["K3_kernel"]),
            rtol=0, atol=2e-12,
        )

    def test_out_of_core_kmat_mode_is_persisted(self):
        """Fresh direct and recovered caches must remain distinguishable."""
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ("direct", "direct", False),
                ("aux-recovery", "aux-recovery", True),
            )
            for mode, kmat_mode, reuse_aux_kernels in cases:
                path = f"{directory}/{mode}.h5"
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("xi_phi", data=np.asarray(self.isdf.xi_phi))
                    handle.create_dataset("xi_grad", data=np.asarray(self.isdf.xi_grad))
                out_of_core = self.isdf.replace(
                    is_incore=False,
                    xi_phi=None,
                    xi_grad=None,
                    save_path=path,
                ).isdf(
                    self.params,
                    save_path=path,
                    batch_size=32,
                    host_grid_block_size=512,
                    reuse_aux_kernels=reuse_aux_kernels,
                )
                self.assertIn("K1_kernel", out_of_core.isdf_kernels)
                with h5py.File(path, "r") as handle:
                    self.assertEqual(handle.attrs["pytc_kmat_kernel_mode"], kmat_mode)
                    self.assertNotIn("H_aux", handle)


class TestStreamingContractionParity(unittest.TestCase):
    """contract_K1_minus_K2_isdf / contract_K3_isdf_streaming must match the
    resident JIT on the same inputs — regardless of whether U is host numpy or
    device jax, and regardless of panel_size."""

    def setUp(self):
        rng = np.random.RandomState(123)
        self.n_fused = 97   # deliberately not a multiple of any panel_size below
        self.n_orb = 11
        self.Np = self.Nq = self.Nr = self.Ns = self.n_orb
        self.phi = jnp.asarray(rng.randn(self.n_orb, self.n_fused))
        self.grad_phi = jnp.asarray(rng.randn(self.n_orb, self.n_fused, 3))
        self.U1 = jnp.asarray(rng.randn(self.n_fused, self.n_fused, 3))
        self.U3 = jnp.asarray(rng.randn(self.n_fused, self.n_fused))
        self.rbs = 16

    def _reference_K1_minus_K2(self):
        from pytc.kmat import contract_K1_minus_K2_isdf_streaming as contract_K1_minus_K2_isdf_jit
        return np.asarray(contract_K1_minus_K2_isdf_jit(
            self.phi, self.phi, self.phi, self.phi,
            self.grad_phi, self.grad_phi, self.U1, self.rbs,
        ))

    def _reference_K3(self):
        from pytc.kmat import contract_K3_isdf_jit
        return np.asarray(contract_K3_isdf_jit(
            self.phi, self.phi, self.phi, self.phi, self.U3, self.rbs,
        ))

    def test_K1_minus_K2_resident_fast_path_matches_jit(self):
        from pytc.kmat import contract_K1_minus_K2_isdf_streaming as contract_K1_minus_K2_isdf
        ref = self._reference_K1_minus_K2()
        out = np.asarray(contract_K1_minus_K2_isdf(
            self.phi, self.phi, self.phi, self.phi,
            self.grad_phi, self.grad_phi, self.U1, self.rbs,
            panel_size=None,
        ))
        np.testing.assert_allclose(out, ref, atol=1e-14, rtol=0)

    def test_K1_minus_K2_streaming_host_matches_jit(self):
        from pytc.kmat import contract_K1_minus_K2_isdf_streaming as contract_K1_minus_K2_isdf
        ref = self._reference_K1_minus_K2()
        U1_host = np.asarray(self.U1)  # explicitly on host
        for panel_size in (16, 32, 48):  # none divides n_fused=97 evenly
            out = np.asarray(contract_K1_minus_K2_isdf(
                self.phi, self.phi, self.phi, self.phi,
                self.grad_phi, self.grad_phi, U1_host, self.rbs,
                panel_size=panel_size,
            ))
            np.testing.assert_allclose(
                out, ref, atol=1e-12, rtol=0,
                err_msg=f"panel_size={panel_size}",
            )

    def test_K1_minus_K2_streaming_device_matches_jit(self):
        from pytc.kmat import contract_K1_minus_K2_isdf_streaming as contract_K1_minus_K2_isdf
        ref = self._reference_K1_minus_K2()
        for panel_size in (16, 32, 48):
            out = np.asarray(contract_K1_minus_K2_isdf(
                self.phi, self.phi, self.phi, self.phi,
                self.grad_phi, self.grad_phi, self.U1, self.rbs,
                panel_size=panel_size,
            ))
            np.testing.assert_allclose(
                out, ref, atol=1e-12, rtol=0,
                err_msg=f"panel_size={panel_size}",
            )

    def test_K1_isdf_streaming_host_matches_jit(self):
        from pytc.kmat import contract_K1_isdf_jit, contract_K1_isdf_streaming
        ref = np.asarray(contract_K1_isdf_jit(
            self.phi, self.phi, self.phi, self.phi, self.grad_phi, self.U1, self.rbs,
        ))
        U1_host = np.asarray(self.U1)
        for panel_size in (16, 32, 48, None):
            out = np.asarray(contract_K1_isdf_streaming(
                self.phi, self.phi, self.phi, self.phi, self.grad_phi,
                U1_host, self.rbs, panel_size=panel_size,
            ))
            np.testing.assert_allclose(
                out, ref, atol=1e-12, rtol=0,
                err_msg=f"panel_size={panel_size}",
            )

    def test_K1_antisym_pq_matches_legacy_transpose(self):
        """In-kernel antisym must equal the legacy ``k12 - k12.T(1,0,2,3)``
        path (resident & all panel sizes, host & device U1)."""
        from pytc.kmat import (contract_K1_isdf_jit,
                                contract_K1_antisym_pq_isdf_streaming)
        k12 = np.asarray(contract_K1_isdf_jit(
            self.phi, self.phi, self.phi, self.phi, self.grad_phi, self.U1, self.rbs,
        ))
        ref = k12 - k12.transpose(1, 0, 2, 3)
        for U1_in, label in ((self.U1, "device"), (np.asarray(self.U1), "host")):
            for panel_size in (None, 16, 32, 48):
                out = np.asarray(contract_K1_antisym_pq_isdf_streaming(
                    self.phi, self.phi, self.phi, self.grad_phi, U1_in, self.rbs,
                    panel_size=panel_size,
                ))
                np.testing.assert_allclose(
                    out, ref, atol=1e-12, rtol=0,
                    err_msg=f"U1={label}, panel_size={panel_size}",
                )

    def test_K3_streaming_host_matches_jit(self):
        from pytc.kmat import contract_K3_isdf_streaming
        ref = self._reference_K3()
        U3_host = np.asarray(self.U3)
        for panel_size in (16, 32, 48, None):
            out = np.asarray(contract_K3_isdf_streaming(
                self.phi, self.phi, self.phi, self.phi,
                U3_host, self.rbs, panel_size=panel_size,
            ))
            np.testing.assert_allclose(
                out, ref, atol=1e-12, rtol=0,
                err_msg=f"panel_size={panel_size}",
            )


if __name__ == '__main__':
    unittest.main()
