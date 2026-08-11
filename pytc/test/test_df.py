import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.df import isdf_decompose
from pytc.integrals.tc import TC

jax.config.update("jax_enable_x64", True)

class TestISDFReconstruction(unittest.TestCase):
    def setUp(self):
        atom = []
        for i in range(4):
            atom.append(f'H 0 0 {i*1.4}')
        self.mol = gto.M(atom=atom, basis='ccpvtz', unit='Bohr', verbose=0)
        self.mf = scf.RHF(self.mol).run()
        
        # We use a dummy Jastrow factor as we only need phi and grad_phi
        from pytc.jastrow.rexp import REXP
        self.tc = TC.from_pyscf(self.mf, REXP(), grid_lvl=1)
        
        self.phi = self.tc.phi
        self.grad_phi = self.tc.grad_phi
        self.weights = self.tc.weights
        
        print(f"\nSystem: H4, Basis: ccpvdz, Grid size: {self.phi.shape[1]}")

    def test_reconstruction_convergence(self):
        """Verify that phi and grad_phi overlaps are reconstructed accurately and converge with rank."""
        n_orb, n_grid = self.phi.shape
        
        # Exact overlaps on grid
        # S_ij = sum_g w_g phi_i(g) phi_j(g)
        phi_weighted = self.phi * self.weights[None, :]
        S_exact = jnp.dot(phi_weighted, self.phi.T)
        
        # G_ij,c = sum_g w_g grad_phi_i,c(g) phi_j(g)
        G_exact = jnp.einsum('igc,g,jg->ijc', self.grad_phi, self.weights, self.phi)
        
        ranks = [400, 600, 800]
        
        print(f"\n{'Rank':<10} {'S Rel Error':<15} {'S Max Abs':<15} {'G Rel Error':<15} {'G Max Abs':<15}")
        print("-" * 75)
        
        prev_S_error = float('inf')
        prev_G_error = float('inf')
        
        for n_rank in ranks:
            phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, _ = isdf_decompose(
                self.phi, self.grad_phi, n_rank, n_rank, weights=self.weights, rcond=1e-14, is_incore=True
            )
            
            # Reconstruct overlaps
            # S_reconst_ij = sum_m C_phi_ij,m * (sum_g xi_phi_m,g * w_g)
            # C_phi_ij,m = phi_i,m * phi_j,m
            xi_phi_weighted_sum = jnp.dot(xi_phi, self.weights)
            C_phi = jnp.einsum('pm,qm->pqm', phi_piv, phi_piv).reshape(-1, len(pivots))
            S_reconst = jnp.dot(C_phi, xi_phi_weighted_sum).reshape(n_orb, n_orb)
            
            # G_reconst_ij,c = sum_m C_grad_ij,m,c * (sum_g xi_grad_m,g,c * w_g)
            # xi_grad is (n_fused, n_grid, 3)
            xi_grad_weighted_sum = jnp.einsum('mgc,g->mc', xi_grad, self.weights)
            # C_grad is (n_orb^2, n_fused, 3)
            # C_grad_ij,m,c = grad_phi_i,m,c * phi_j,m
            C_grad = jnp.einsum('pmc,qm->pqmc', grad_phi_piv, phi_piv).reshape(-1, len(pivots), 3)
            G_reconst = jnp.einsum('nmc,mc->nc', C_grad, xi_grad_weighted_sum).reshape(n_orb, n_orb, 3)
            
            S_error = jnp.linalg.norm(S_reconst - S_exact) / jnp.linalg.norm(S_exact)
            G_error = jnp.linalg.norm(G_reconst - G_exact) / jnp.linalg.norm(G_exact)
            
            S_max_abs = jnp.max(jnp.abs(S_reconst - S_exact))
            G_max_abs = jnp.max(jnp.abs(G_reconst - G_exact))
            
            print(f"{n_rank:<10} {S_error:<15.2e} {S_max_abs:<15.2e} {G_error:<15.2e} {G_max_abs:<15.2e}")
            
            if n_rank > ranks[0]:
                if S_error > 1e-12:
                    self.assertLess(S_error, prev_S_error, f"S error did not decrease at rank {n_rank}")
                if G_error > 1e-12:
                    self.assertLess(G_error, prev_G_error, f"G error did not decrease at rank {n_rank}")
            
            prev_S_error = S_error
            prev_G_error = G_error
            
        self.assertLess(S_error, 1e-4, f"Final S reconstruction error {S_error} is too high")
        self.assertLess(G_error, 1e-3, f"Final G reconstruction error {G_error} is too high")

class TestISDFMultiDeviceSharding(unittest.TestCase):
    """Run the ISDF grid-batch solve under 1 vs N simulated CPU devices and
    verify the sharded output matches the single-device output.

    XLA_FLAGS must be set before jax import, so the work is done in a subprocess.
    OMP and XLA Eigen threading are pinned to 1 so device-level parallelism is
    the only axis of variation. Timings are reported for diagnostic purposes;
    we do not assert a speedup because simulated multi-CPU shares physical
    cores — real speedup requires actual multi-GPU hardware.
    """

    CHILD = r"""
import os, sys, time, json
import jax, jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update('jax_enable_x64', True)
from pytc.df import (prepare_normal_equations_solver,
                     solve_normal_equations_batch_prepared)

# Synthetic problem sized so the grid solve is the dominant cost.
n_orb, n_rank, n_grid, batch = 80, 900, 32768, 2048
rng = np.random.default_rng(0)
phi = jnp.asarray(rng.standard_normal((n_orb, n_grid)))
phi_piv = jnp.asarray(rng.standard_normal((n_orb, n_rank)))

chol, lower = prepare_normal_equations_solver(phi_piv, phi_piv)

devices = jax.devices()
n_dev = len(devices)
use_shard = n_dev > 1
if use_shard:
    mesh = Mesh(np.array(devices), ('g',))
    shard = NamedSharding(mesh, P(None, 'g'))
    repl = NamedSharding(mesh, P())
    chol = jax.device_put(chol, repl)
    phi_piv_d = jax.device_put(phi_piv, repl)
else:
    phi_piv_d = phi_piv

def run_loop():
    last = None
    for g in range(0, n_grid, batch):
        b = phi[:, g:g + batch]
        if use_shard:
            b = jax.device_put(b, shard)
        last = solve_normal_equations_batch_prepared(chol, lower, phi_piv_d, phi_piv_d, b, b)
    jax.block_until_ready(last)
    return last

# Warm-up JIT.
warm = run_loop()

# Collect all batches for a checksum we can compare across runs.
outs = []
t0 = time.perf_counter()
for g in range(0, n_grid, batch):
    b = phi[:, g:g + batch]
    if use_shard:
        b = jax.device_put(b, shard)
    x = solve_normal_equations_batch_prepared(chol, lower, phi_piv_d, phi_piv_d, b, b)
    outs.append(x)
jax.block_until_ready(outs[-1])
dt = time.perf_counter() - t0

full = jnp.concatenate(outs, axis=1)
print('RESULT ' + json.dumps({
    'n_devices': n_dev,
    'dt': dt,
    'sum_sq': float(jnp.sum(full ** 2)),
}))
"""

    def _run_child(self, n_devices):
        import os, sys, json, subprocess
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = "1"
        # Force serial intra-op threads so device-level parallelism is the only
        # source of speedup (otherwise XLA's Eigen threadpool oversubscribes).
        env["XLA_FLAGS"] = (f"--xla_force_host_platform_device_count={n_devices} "
                            f"--xla_cpu_multi_thread_eigen=false")
        r = subprocess.run([sys.executable, "-c", self.CHILD],
                           env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            self.fail(f"child (n_devices={n_devices}) failed:\n"
                      f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        self.fail(f"no RESULT line in child stdout:\n{r.stdout}")

    def test_multidevice_parity(self):
        import multiprocessing
        if multiprocessing.cpu_count() < 4:
            self.skipTest("need 4+ CPUs to simulate multi-device")

        r1 = self._run_child(1)
        r4 = self._run_child(4)

        speedup = r1['dt'] / r4['dt']
        print(f"\nISDF grid solve  1 dev: {r1['dt']:.2f}s  "
              f"4 dev: {r4['dt']:.2f}s  speedup: {speedup:.2f}x")

        # Correctness parity: sharded output must match single-device bit-for-bit
        # within float64 rounding. This is the real regression check.
        rel = abs(r1['sum_sq'] - r4['sum_sq']) / max(abs(r1['sum_sq']), 1e-30)
        self.assertLess(rel, 1e-8, f"sum_sq drifted under sharding: rel diff {rel:.2e}")

        # We do NOT assert on wall-clock speedup: simulated multi-CPU shares L1/L2
        # caches and physical cores, so overhead dominates at test-scale problems.
        # Real speedup requires actual multi-GPU hardware — the timings printed
        # above are informational only.


if __name__ == '__main__':
    unittest.main()
