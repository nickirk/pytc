
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf, lib, cc

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import xtc_ccsd
from pytc.solver import jax_xtc_ccsd
import logging


jax.config.update("jax_enable_x64", True)

import h5py
import os

def save_eris_to_h5(eris, path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('fock', data=np.asarray(eris.fock))
        f.create_dataset('mo_energy', data=np.asarray(eris.mo_energy))
        f.create_dataset('e_core', data=np.float64(eris.e_core))
        for block in ['oooo', 'ovoo', 'ooov', 'vooo', 'ovov', 'vovo', 'ovvo', 'voov', 'oovv', 'vvoo', 'ovvv', 'vvov', 'vovv', 'vvvv']:
            val = getattr(eris, block, None)
            if val is not None:
                f.create_dataset(block, data=np.asarray(val))

def load_eris_from_h5(path, mol):
    from pyscf.cc import rccsd
    eris = xtc_ccsd._ChemistsERIs(mol)
    with h5py.File(path, 'r') as f:
        eris.fock = np.array(f['fock'])
        eris.mo_energy = np.array(f['mo_energy'])
        eris.e_core = float(f['e_core'][()])
        for block in ['oooo', 'ovoo', 'ooov', 'vooo', 'ovov', 'vovo', 'ovvo', 'voov', 'oovv', 'vvoo', 'ovvv', 'vvov', 'vovv', 'vvvv']:
            if block in f:
                setattr(eris, block, np.array(f[block]))
            else:
                setattr(eris, block, None)
    eris.nocc = np.sum(mol.nelec) // 2 # Simple closed shell assumption for test
    return eris

class TestXTCCCSD(unittest.TestCase):
    def setUp(self):

        self.mol = gto.M(
            atom='C 0 0 0; O 0 0 1.128',
            basis='sto-6g',
            verbose=4
        )
        self.mf = scf.RHF(self.mol).run()
        
        self.jastrow = rexp.REXP()
        self.jastrow_params = {'alpha': jnp.array([0.5])}
        
        self.xtc_obj = xtc.XTC.from_pyscf(self.mf, self.jastrow, grid_lvl=1)
        
        self.n_rank = self.xtc_obj.n_orb * 12 # Sufficiently high rank
        self.isdf_xtc = xtc.ISDFXTC.from_xtc(self.xtc_obj, n_rank=self.n_rank, save_path="isdf_xtc_ccsd_test.h5")
        self.isdf_xtc = self.isdf_xtc.isdf(self.jastrow_params) # Precompute kernels

    def test_rccsd_energy(self):
        print("\nRunning Reference Exact XTC CCSD...")
        exact_h5 = "eris_exact_test.h5"
        if os.path.exists(exact_h5):
            print(f"Loading exact ERIs from {exact_h5}")
            eris_exact = load_eris_from_h5(exact_h5, self.mol)
        else:
            eris_exact = self.xtc_obj.make_eris(self.mf, self.jastrow_params)
            print(f"Saving exact ERIs to {exact_h5}")
            save_eris_to_h5(eris_exact, exact_h5)
            
        cc_exact = xtc_ccsd.RCCSD(self.mf, self.xtc_obj, self.jastrow_params)
        
        print("\nBuilding New ISDF-XTC-RCCSD ERIs...")
        cc_new = jax_xtc_ccsd.RCCSD(self.mf, self.isdf_xtc, self.jastrow_params)
        eris_new = cc_new.ao2mo()
        
        print("\nComparing Intermediates...")
        # We manually compute one step of update_amps to compare
        nocc = self.mf.mol.nelec[0]
        nmo = self.xtc_obj.n_orb
        nvir = nmo - nocc
        
        # Initialize with MP2-like amplitudes to check non-zero terms
        eris_ref = eris_exact
        mo_e = eris_ref.mo_energy
        eia = mo_e[:nocc, None] - mo_e[None, nocc:]
        eijab = eia[:, None, :, None] + eia[None, :, None, :]
        
        ovov_ref = np.asarray(eris_ref.ovov)
        t2 = ovov_ref.transpose(0,2,1,3).conj() / eijab
        t1 = np.zeros((nocc, nvir)) # Start with t1=0
        
        cc_ref = xtc_ccsd.RCCSD(self.mf, self.xtc_obj, self.jastrow_params)
        
        import jax.numpy as jnp
        fock_jax = jnp.asarray(eris_new.fock)
        eris_ovvo_jax = jnp.asarray(eris_new.ovvo)
        eris_oovv_jax = jnp.asarray(eris_new.oovv)
        eris_ovoo_jax = jnp.asarray(eris_new.ovoo)
        eris_ovov_jax = jnp.asarray(eris_new.ovov)
        eris_oooo_jax = jnp.asarray(eris_new.oooo)
        
        t1_jax = jnp.asarray(t1)
        t2_jax = jnp.asarray(t2)
        
        from pyscf.cc import rintermediates as imd
        Foo_ref = imd.cc_Foo(t1, t2, eris_ref)
        Fvv_ref = imd.cc_Fvv(t1, t2, eris_ref)
        Fov_ref = imd.cc_Fov(t1, t2, eris_ref)
        
        Foo_jax = jax_xtc_ccsd._jax_cc_Foo(t1_jax, t2_jax, fock_jax, eris_ovov_jax)
        Fvv_jax = jax_xtc_ccsd._jax_cc_Fvv(t1_jax, t2_jax, fock_jax, eris_ovov_jax)
        Fov_jax = jax_xtc_ccsd._jax_cc_Fov(t1_jax, fock_jax, eris_ovov_jax)
        
        print(f"Foo Difference Norm: {np.linalg.norm(Foo_ref - Foo_jax)}")
        print(f"Fvv Difference Norm: {np.linalg.norm(Fvv_ref - Fvv_jax)}")
        print(f"Fov Difference Norm: {np.linalg.norm(Fov_ref - Fov_jax)}")
        
        Woooo_jax = jax_xtc_ccsd._jax_cc_Woooo(t1_jax, t2_jax, eris_oooo_jax, eris_ovov_jax, eris_ovoo_jax)
        Woooo_ref = imd.cc_Woooo(t1, t2, eris_ref)
        print(f"Woooo Difference Norm: {np.linalg.norm(Woooo_ref - Woooo_jax)}")
        
        mo_e = eris_ref.mo_energy
        Loo_ref = imd.Loo(t1, t2, eris_ref)
        Loo_ref[np.diag_indices(nocc)] -= mo_e[:nocc]
        
        Loo_jax = jax_xtc_ccsd._jax_Loo(t1_jax, t2_jax, fock_jax, eris_ovov_jax, eris_ovoo_jax)
        Loo_jax = Loo_jax.at[jnp.diag_indices(nocc)].add(-mo_e[:nocc])
        
        print(f"Loo (shifted) Difference Norm: {np.linalg.norm(Loo_ref - Loo_jax)}")
        
        Lvv_ref = imd.cc_Fvv(t1, t2, eris_ref) - np.einsum('kc,ka->ac', eris_ref.fock[:nocc, nocc:], t1)
        Lvv_ref[np.diag_indices(nvir)] -= mo_e[nocc:]
        
        Lvv_jax = jax_xtc_ccsd._jax_Lvv(t1_jax, t2_jax, fock_jax, eris_ovov_jax, eris_ovvv_all=None) # Pass None to skip ovvv
        Lvv_jax = Lvv_jax.at[jnp.diag_indices(nvir)].add(-mo_e[nocc:])
        
        print(f"Lvv (non-ovvv, shifted) Difference Norm: {np.linalg.norm(Lvv_ref - Lvv_jax)}")
        
        # Ref Wvoov is full. We can extract non-ovvv part from ref or compute full jax
        
        Wvoov_jax = eris_ovvo_jax.transpose(2,0,3,1)
        Wvoov_jax -= jnp.einsum('kcli,la->akic', eris_ovoo_jax, t1_jax)
        Wvoov_jax -= 0.5 * jnp.einsum('ldkc,ilda->akic', eris_ovov_jax, t2_jax)
        Wvoov_jax -= 0.5 * jnp.einsum('lckd,ilad->akic', eris_ovov_jax, t2_jax)
        Wvoov_jax -=       jnp.einsum('ldkc,id,la->akic', eris_ovov_jax, t1_jax, t1_jax)
        Wvoov_jax +=       jnp.einsum('ldkc,ilad->akic', eris_ovov_jax, t2_jax)
        
        Wvovo_jax = eris_oovv_jax.transpose(2,0,3,1)
        Wvovo_jax -= jnp.einsum('lcki,la->akci', eris_ovoo_jax, t1_jax)
        Wvovo_jax -= 0.5 * jnp.einsum('lckd,ilda->akci', eris_ovov_jax, t2_jax)
        Wvovo_jax -=       jnp.einsum('lckd,id,la->akci', eris_ovov_jax, t1_jax, t1_jax)

        ovvv_all_jax = jnp.asarray(eris_new.ovvv)
        tau_jax = t2_jax + jnp.einsum('ia,jb->ijab', t1_jax, t1_jax)
        
        _, Lvv_ovvv, Wvoov_ovvv, Wvovo_ovvv, _, _ = jax_xtc_ccsd.kernel_process_ovvv_block(ovvv_all_jax, t1_jax, t2_jax, tau_jax)
        
        Wvoov_jax_full = Wvoov_jax + Wvoov_ovvv
        Wvovo_jax_full = Wvovo_jax + Wvovo_ovvv
        
        Wvoov_ref = imd.cc_Wvoov(t1, t2, eris_ref)
        Wvovo_ref = imd.cc_Wvovo(t1, t2, eris_ref)
        
        print(f"Wvoov Full Difference Norm: {np.linalg.norm(Wvoov_ref - Wvoov_jax_full)}")
        print(f"Wvovo Full Difference Norm: {np.linalg.norm(Wvovo_ref - Wvovo_jax_full)}")
        
        vovo_jax_trans = eris_new.vovo.transpose(1,3,0,2)
        vovo_ref_trans = eris_ref.vovo.transpose(1,3,0,2)
        print(f"vovo Transpose (JAX vs Ref) Difference Norm: {np.linalg.norm(vovo_ref_trans - vovo_jax_trans)}")
        
        ovov_ref_trans = eris_ref.ovov.transpose(0,2,1,3)
        print(f"Ref Internals (ovov vs vovo) Difference Norm: {np.linalg.norm(ovov_ref_trans - vovo_ref_trans)}")
        
        ovov_jax_trans = eris_new.ovov.transpose(0,2,1,3)
        print(f"JAX Internals (ovov vs vovo) Difference Norm: {np.linalg.norm(ovov_jax_trans - vovo_jax_trans)}")
        
        ovoo_t2_ref = -2*lib.einsum('lcki,klac->ia', eris_ref.ovoo, t2)
        ovoo_t2_ref += lib.einsum('kcli,klac->ia', eris_ref.ovoo, t2)
        
        
        ovvv_t1_ref = 2*lib.einsum('kdac,ikcd->ia', eris_ref.get_ovvv(), t2)
        ovvv_t1_ref -= lib.einsum('kcad,ikcd->ia', eris_ref.get_ovvv(), t2)
        print(f"Ref ovvv*t2 Norm: {np.linalg.norm(ovvv_t1_ref)}")
        
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        woooo_tau_ref = lib.einsum('klij,klab->ijab', Woooo_ref, tau)
        print(f"Ref Woooo*tau Norm: {np.linalg.norm(woooo_tau_ref)}")

        print("\nRunning Reference Update Amps (Cycle 1)...")
        t1new_ref, t2new_ref = cc_ref.update_amps(t1, t2, eris_ref)
        
        print("\nRunning JAX Update Amps (Cycle 1)...")
        t1new_jax_full, t2new_jax_full = cc_new.update_amps(t1, t2, eris_new)
        
        self.assertLess(np.linalg.norm(t1new_ref - t1new_jax_full), 1e-6)
        self.assertLess(np.linalg.norm(t2new_ref - t2new_jax_full), 1e-6)

        
        # Compare blocks BEFORE zeroing vvvv
        fock_diff = np.linalg.norm(eris_new.fock - eris_exact.fock)
        print(f"Fock Difference Norm: {fock_diff}")
    
        try:
            if eris_new.oooo.shape == eris_exact.oooo.shape:
                oooo_diff = np.linalg.norm(eris_new.oooo - eris_exact.oooo)
                print(f"OOOO Difference Norm: {oooo_diff}")
            else:
                print(f"OOOO Shape mismatch")
        except Exception as e:
            print(f"OOOO check failed: {e}")
        
        try:
            if eris_new.oovv.shape == eris_exact.oovv.shape:
                oovv_diff = np.linalg.norm(eris_new.oovv - eris_exact.oovv)
                print(f"OOVV Difference Norm: {oovv_diff}")
            else:
                print(f"OOVV Shape mismatch")
        except Exception as e:
            print(f"OOVV check failed: {e}")

        try:
            if eris_new.ovov.shape == eris_exact.ovov.shape:
                ovov_diff = np.linalg.norm(eris_new.ovov - eris_exact.ovov)
                print(f"OVOV Difference Norm: {ovov_diff}")
            else:
                print(f"OVOV Shape mismatch")
        except Exception as e:
            print(f"OVOV check failed: {e}")
        
        try:
            if eris_new.ovoo.shape == eris_exact.ovoo.shape:
                ovoo_diff = np.linalg.norm(eris_new.ovoo - eris_exact.ovoo)
                print(f"OVOO Difference Norm: {ovoo_diff}")
            else:
                print(f"OVOO Shape mismatch")
        except Exception as e:
            print(f"OVOO check failed: {e}")

        if eris_exact.ovvv is not None:
            def unpack_if_needed(ov):
                if ov.ndim == 3: # packed (nocc, nvir, pair)
                    nocc, nvir, pair = ov.shape
                    unpacked = np.zeros((nocc, nvir, nvir, nvir))
                    idx = np.tril_indices(nvir)
                    unpacked[:, :, idx[0], idx[1]] = ov
                    unpacked[:, :, idx[1], idx[0]] = ov
                    return unpacked
                return ov # already unpacked (nocc, nvir, nvir, nvir)
            
            ovvv_ref = unpack_if_needed(eris_exact.ovvv)
            ovvv_new = unpack_if_needed(eris_new.ovvv)
            
            if ovvv_ref.shape == ovvv_new.shape:
                ovvv_diff = np.linalg.norm(ovvv_ref - ovvv_new)
                print(f"OVVV Difference Norm: {ovvv_diff}")
                print(f"OVVV Ref max: {np.max(np.abs(ovvv_ref))}")
                print(f"OVVV New max: {np.max(np.abs(ovvv_new))}")
            else:
                print(f"OVVV shapes still differ: {ovvv_new.shape} vs {ovvv_ref.shape}")
        
        if eris_exact.vvvv is not None and eris_new.vvvv is not None:
            # Save copies before zeroing for comparison
            # Use np.array() instead of .copy() to handle both numpy arrays and HDF5 Datasets
            vvvv_ref_saved = np.array(eris_exact.vvvv) if hasattr(eris_exact, 'vvvv') else None
            vvvv_new_saved = np.array(eris_new.vvvv) if hasattr(eris_new, 'vvvv') else None
            if vvvv_ref_saved is not None and vvvv_new_saved is not None:
                vvvv_diff = np.linalg.norm(vvvv_ref_saved - vvvv_new_saved)
                print(f"VVVV Difference Norm: {vvvv_diff}")
                print(f"VVVV Ref max: {np.max(np.abs(vvvv_ref_saved))}")
                print(f"VVVV New max: {np.max(np.abs(vvvv_new_saved))}")

        print("\nRunning Reference CCSD (Exact XTC)...")
        e_ref, t1_ref, t2_ref = cc_exact.kernel(eris=eris_exact)
        print(f"Reference Correlation Energy: {e_ref}")
        
        print("\nRunning New ISDF-XTC-CCSD...")
        e_new, t1_new, t2_new = cc_new.kernel(eris=eris_new)

        
        print(f"New Correlation Energy: {e_new}")
        
        error = abs(e_new - e_ref)
        print(f"Energy Difference: {error}")
        
        self.assertLess(error, 1e-6, "Correlation energy mismatch > 1e-6 Ha")
        
        t1_diff = np.linalg.norm(t1_new - t1_ref)
        self.assertLess(t1_diff, 1e-6, "T1 amplitude mismatch")

    def test_hermitian_limit(self):
        print("\nTesting Hermitian Limit (Large Alpha)...")
        # Use large alpha to make Jastrow negligible
        large_alpha_params = {'alpha': jnp.array([1000.0])}
        
        # Exact XTC with large alpha should match PySCF standard
        cc_xtc = xtc_ccsd.RCCSD(self.mf, self.xtc_obj, large_alpha_params)
        
        limit_h5 = "eris_limit_test.h5"
        if os.path.exists(limit_h5):
            print(f"Loading limit ERIs from {limit_h5}")
            eris_xtc = load_eris_from_h5(limit_h5, self.mol)
        else:
            eris_xtc = self.xtc_obj.make_eris(self.mf, large_alpha_params)
            print(f"Saving limit ERIs to {limit_h5}")
            save_eris_to_h5(eris_xtc, limit_h5)
            
        e_xtc, _, _ = cc_xtc.kernel(eris=eris_xtc)
        
        cc_std = cc.rccsd.RCCSD(self.mf)
        e_std, _, _ = cc_std.kernel()
        
        print(f"XTC (Large Alpha) Energy: {e_xtc}")
        print(f"PySCF Standard Energy: {e_std}")
        
        print("\nComparing mo_energy:")
        print(f"XTC mo_energy: {eris_xtc.mo_energy}")
        print(f"STD mo_energy: {self.mf.mo_energy}")

        
        error = abs(e_xtc - e_std)

        print(f"Energy Difference (Hermitian Limit): {error}")
        
        self.assertLess(error, 1e-6, "Hermitian limit agreement failed")


    def tearDown(self):
        import os
        for f in ["isdf_xtc_ccsd_test.h5", "eris_exact_test.h5", "eris_limit_test.h5"]:
            if os.path.exists(f):
                os.remove(f)

class TestShouldForceHostAccumulators(unittest.TestCase):
    """Regression tests for ``_should_force_host_accumulators``.

    The OVVV/VOVV pipelines below ``_update_amps`` cannot keep
    accumulators GPU-resident under three independent conditions:
    multi-GPU, HDF5-backed ovvv, or HDF5-backed vovv.  The original
    early override only checked the multi-GPU case, so a single-GPU run
    on a system large enough to spill ovvv to disk hit a misleading
    AssertionError further down::

        AssertionError: use_gpu_acc must be False for the HDF5-backed
        OVVV multi-GPU path; set by _n_devices_local > 1 check above

    The helper centralises the predicate so the inline override and
    these tests stay in sync.  Each combination of
    ``(n_devices_local, ovvv in RAM, vovv in RAM)`` is checked
    explicitly so future edits cannot silently weaken any branch.
    """

    class _StubEris:
        """Minimal stand-in for ``_ChemistsERIs`` — only the attributes
        the helper inspects need to be set."""
        def __init__(self, ovvv, vovv):
            self.ovvv = ovvv
            self.vovv = vovv

    def setUp(self):
        # An in-RAM ndarray and an HDF5-like sentinel that fails the
        # ``isinstance(_, np.ndarray)`` check.  We don't need a real
        # ``h5py.Dataset`` — the helper only looks at the type.
        self._in_ram = np.zeros((2, 2, 2, 2))
        self._on_disk = object()  # any non-ndarray value triggers the override

    def test_single_gpu_in_ram_keeps_gpu_accumulators(self):
        """The only configuration where GPU-resident accumulators are safe."""
        eris = self._StubEris(self._in_ram, self._in_ram)
        self.assertFalse(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=1)
        )

    def test_multi_gpu_in_ram_forces_host(self):
        """Multi-GPU + in-RAM ovvv/vovv: round-robin needs a shared host buffer."""
        eris = self._StubEris(self._in_ram, self._in_ram)
        self.assertTrue(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=2)
        )

    def test_single_gpu_hdf5_ovvv_forces_host(self):
        """Regression: 1 GPU + HDF5-backed ovvv must force host
        accumulators.  Before the fix this passed the early check
        (n_devices_local==1) and crashed at the HDF5-OVVV branch."""
        eris = self._StubEris(self._on_disk, self._in_ram)
        self.assertTrue(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=1)
        )

    def test_single_gpu_hdf5_vovv_forces_host(self):
        """Same regression on the VOVV side: 1 GPU + HDF5-backed vovv."""
        eris = self._StubEris(self._in_ram, self._on_disk)
        self.assertTrue(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=1)
        )

    def test_single_gpu_both_hdf5_forces_host(self):
        """The realistic large-system case: both ovvv and vovv on disk."""
        eris = self._StubEris(self._on_disk, self._on_disk)
        self.assertTrue(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=1)
        )

    def test_multi_gpu_hdf5_both_forces_host(self):
        """Multi-GPU + both HDF5: every condition fires; sanity check the OR."""
        eris = self._StubEris(self._on_disk, self._on_disk)
        self.assertTrue(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=4)
        )

    def test_zero_devices_treated_as_single_device(self):
        """``n_devices_local == 0`` (CPU-only fallback) is not multi-GPU
        and must not by itself force host accumulators when ovvv/vovv
        are in RAM."""
        eris = self._StubEris(self._in_ram, self._in_ram)
        self.assertFalse(
            jax_xtc_ccsd._should_force_host_accumulators(eris, n_devices_local=0)
        )


if __name__ == "__main__":
    unittest.main()
