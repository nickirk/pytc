"""Benchmark-style comparisons for Boys-Handy implementations.

These tests are intended to make the performance delta between the original
autodiff ``BoysHandy`` and ``BoysHandyAnalytical`` visible on the local
machine. They print timings and verify numerical agreement, but keep runtime
bounded by benchmarking the core ``compute_jastrow_terms`` hotspot directly.
"""

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.ansatz.det import SlaterDet
from pytc.ansatz.sj import SlaterJastrow
from pytc.jastrow import CompositeJastrow, NuclearCusp, BoysHandy, BoysHandyAnalytical
from pytc.vmc.hamiltonian import compute_jastrow_terms
from pytc.vmc.walker import initialize_walkers

jax.config.update("jax_enable_x64", True)


def make_h_chain(n_atoms, dist=1.0):
    return "; ".join(f"H 0 0 {i * dist}" for i in range(n_atoms))


def build_chain_system(n_atoms=20, basis="sto-3g"):
    mol = gto.Mole()
    mol.atom = make_h_chain(n_atoms)
    mol.basis = basis
    mol.unit = "Bohr"
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    det = SlaterDet.create(mol, mf.mo_coeff)
    return mol, det


def make_bh_ansatz(mol, det, bh_cls):
    jastrow = CompositeJastrow.create([
        NuclearCusp.create(mol),
        bh_cls.create(mol),
    ])
    sj = SlaterJastrow.create(mol, jastrow, [det])
    params = [jastrow.init_params(), jnp.ones(1)]
    return sj, params


class TestBoysHandyBenchmark(unittest.TestCase):
    def setUp(self):
        self.key = random.PRNGKey(0)
        self.systems = {}
        for label, n_atoms, basis in [
            ("H20/STO-3G", 20, "sto-3g"),
            ("H40/STO-3G", 40, "sto-3g"),
        ]:
            mol, det = build_chain_system(n_atoms=n_atoms, basis=basis)
            walkers = initialize_walkers(det, 1, key=self.key)
            sj_bh, params_bh = make_bh_ansatz(mol, det, BoysHandy)
            sj_bha, params_bha = make_bh_ansatz(mol, det, BoysHandyAnalytical)
            batch_ansatz_bh = jax.vmap(lambda w, p: sj_bh(w, p), in_axes=(0, None))
            _, walkers_bh = batch_ansatz_bh(walkers, params_bh)
            self.systems[label] = {
                "positions": walkers_bh.positions[0],
                "sj_bh": sj_bh,
                "params_bh": params_bh,
                "sj_bha": sj_bha,
                "params_bha": params_bha,
            }

    def _benchmark_terms(self, sj, params, positions, warm_repeats=5):
        fn = jax.jit(lambda pos: compute_jastrow_terms(sj, pos, params[0]))

        jax.clear_caches()
        t0 = time.time()
        grad, lap = fn(positions)
        jax.block_until_ready(grad)
        jax.block_until_ready(lap)
        first = time.time() - t0

        warm_times = []
        for _ in range(warm_repeats):
            t1 = time.time()
            grad, lap = fn(positions)
            jax.block_until_ready(grad)
            jax.block_until_ready(lap)
            warm_times.append(time.time() - t1)

        return first, float(np.mean(warm_times)), np.array(grad), np.array(lap)

    def test_compute_jastrow_terms_speedup(self):
        for label, system in self.systems.items():
            first_bh, warm_bh, grad_bh, lap_bh = self._benchmark_terms(
                system["sj_bh"], system["params_bh"], system["positions"]
            )
            first_bha, warm_bha, grad_bha, lap_bha = self._benchmark_terms(
                system["sj_bha"], system["params_bha"], system["positions"]
            )

            np.testing.assert_allclose(grad_bha, grad_bh, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(lap_bha, lap_bh, rtol=1e-6, atol=1e-6)

            compile_speedup = first_bh / first_bha if first_bha > 0 else np.inf
            warm_speedup = warm_bh / warm_bha if warm_bha > 0 else np.inf

            print(
                f"\n{label} BH first-call: {first_bh:.3f}s | "
                f"BHA: {first_bha:.3f}s | speedup: {compile_speedup:.2f}x"
            )
            print(
                f"{label} BH warm avg:  {warm_bh:.3f}s | "
                f"BHA: {warm_bha:.3f}s | speedup: {warm_speedup:.2f}x"
            )

            self.assertTrue(np.isfinite(first_bh) and np.isfinite(first_bha))
            self.assertTrue(np.isfinite(warm_bh) and np.isfinite(warm_bha))


if __name__ == "__main__":
    unittest.main()
