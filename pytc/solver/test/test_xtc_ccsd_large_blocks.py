import os
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, lib

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import xtc_ccsd


jax.config.update("jax_enable_x64", True)


class TestXTCCCSDLargeBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.save_path = tempfile.mkstemp(
            prefix="isdf_xtc_large_blocks_", suffix=".h5"
        )
        os.close(fd)
        if os.path.exists(cls.save_path):
            os.remove(cls.save_path)

        cls.mol = gto.M(
            atom="O 0 0 0; H 0 0 0.958; H 0.0 0.926 -0.239",
            basis="ccpvdz",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).density_fit().run()
        cls.jastrow = rexp.REXP()
        cls.jastrow_params = {"alpha": jnp.array([0.5])}
        cls.xtc_obj = xtc.XTC.from_pyscf(cls.mf, cls.jastrow, grid_lvl=0)
        cls.n_rank = cls.xtc_obj.n_orb * 4
        cls.isdf_xtc = xtc.ISDFXTC.from_xtc(
            cls.xtc_obj, n_rank=cls.n_rank, save_path=cls.save_path
        )
        cls.isdf_xtc = cls.isdf_xtc.isdf(cls.jastrow_params, save_path=cls.save_path)

        cls.cc = xtc_ccsd.RCCSD(
            cls.mf,
            cls.isdf_xtc,
            cls.jastrow_params,
            on_the_fly_vvvv=True,
            max_memory=32000,
            gpu_max_memory=4096,
        ).density_fit(with_df=cls.mf.with_df)

        cls.nocc = cls.mf.mol.nelectron // 2
        cls.nmo = cls.mf.mo_coeff.shape[1]
        cls.nvir = cls.nmo - cls.nocc
        cls.occ_all = slice(0, cls.nocc)
        cls.vir_all = slice(cls.nocc, cls.nmo)

        eris_ref = xtc_ccsd._ChemistsERIs(cls.cc.mol)
        eris_ref._common_init_(cls.cc, cls.cc.mo_coeff)
        eris_ref.xtc_obj = cls.isdf_xtc
        eris_ref.jastrow_params = cls.jastrow_params
        eris_ref.max_memory = cls.cc.max_memory
        eris_ref.gpu_max_memory = cls.cc.gpu_max_memory

        naux = cls.mf.with_df.get_naoaux()
        _, Lov = xtc_ccsd._init_df_eris(
            eris_ref, cls.mf.with_df, cls.nvir, naux, cls.nocc, cls.nmo, cls.cc.mo_coeff
        )
        cls.Lov_reshaped = Lov.reshape(naux, cls.nocc, cls.nvir)
        cls.L_vv_full = lib.unpack_tril(eris_ref.vvL[:], axis=0)

        tc_ovvv = np.asarray(
            cls.isdf_xtc.get_2b(
                cls.jastrow_params,
                ranges=(cls.occ_all, cls.vir_all, cls.vir_all, cls.vir_all),
            )
        )
        tc_vovv = np.asarray(
            cls.isdf_xtc.get_2b(
                cls.jastrow_params,
                ranges=(cls.vir_all, cls.occ_all, cls.vir_all, cls.vir_all),
            )
        )
        std_ovvv = np.tensordot(cls.Lov_reshaped, cls.L_vv_full, axes=((0,), (2,)))
        std_vovv = std_ovvv.transpose(1, 0, 2, 3)
        cls.ref_ovvv = std_ovvv + tc_ovvv
        cls.ref_vovv = std_vovv + tc_vovv
        eris_ref.close()

        cls.recorded_layouts = []
        real_compute_2b_tile = xtc_ccsd.xtc_mod.compute_2b_tile

        def wrapped_compute_2b_tile(xtc_obj, jastrow_params, ranges,
                                    device=None, panel_size=None, panel_layout="pr"):
            cls.recorded_layouts.append((ranges, panel_layout))
            return real_compute_2b_tile(
                xtc_obj,
                jastrow_params,
                ranges,
                device=device,
                panel_size=panel_size,
                panel_layout=panel_layout,
            )

        with mock.patch.object(
            xtc_ccsd, "resolve_v3o_panel_block_size", return_value=2
        ), mock.patch.object(
            xtc_ccsd.xtc_mod, "compute_2b_tile", side_effect=wrapped_compute_2b_tile
        ):
            cls.eris = cls.cc.ao2mo()

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "eris") and hasattr(cls.eris, "close"):
                cls.eris.close()
        finally:
            if os.path.exists(cls.save_path):
                os.remove(cls.save_path)

    def test_ovvv_matches_public_reference(self):
        np.testing.assert_allclose(
            np.asarray(self.eris.ovvv),
            self.ref_ovvv,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_vovv_matches_public_reference(self):
        np.testing.assert_allclose(
            np.asarray(self.eris.vovv),
            self.ref_vovv,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_large_block_build_uses_expected_panel_layouts(self):
        saw_ovvv_qr = False
        saw_vovv_pr = False
        for ranges, layout in self.recorded_layouts:
            p, q, _, _ = ranges
            p_len = p.stop - p.start
            q_len = q.stop - q.start
            if p.start == 0 and p.stop == self.nocc and q.start == self.nocc:
                saw_ovvv_qr |= (layout == "qr")
            if p.start >= self.nocc and q.start == 0 and q.stop == self.nocc:
                saw_vovv_pr |= (layout == "pr")
        self.assertTrue(saw_ovvv_qr, "ovvv builder never used panel_layout='qr'")
        self.assertTrue(saw_vovv_pr, "vovv builder never used panel_layout='pr'")

    def test_large_block_datasets_are_chunked(self):
        self.assertIsNotNone(self.eris.ovvv.chunks)
        self.assertIsNotNone(self.eris.vovv.chunks)
        self.assertEqual(self.eris.ovvv.chunks[0], self.nocc)
        self.assertEqual(self.eris.ovvv.chunks[2], 2)
        self.assertEqual(self.eris.vovv.chunks[0], 2)
        self.assertEqual(self.eris.vovv.chunks[1], self.nocc)
