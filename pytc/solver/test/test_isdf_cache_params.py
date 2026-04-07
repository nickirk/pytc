import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import bh
from pytc.solver import xtc_ccsd


# Enable float64 for deterministic energies
jax.config.update("jax_enable_x64", True)


def test_isdf_cache_recomputes_for_new_jastrow(tmp_path):
    """Cached ISDF kernels must be recomputed when Jastrow params change."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    jastrow = bh.BoysHandy.create(mol)
    params_a = jastrow.init_params()
    params_b = {
        "b_raw": params_a["b_raw"] * 1.2,
        "d_raw": params_a["d_raw"] * 0.9,
        "c_raw": params_a["c_raw"] * 1.1,
    }

    xtc_obj = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=0)
    n_rank = 2  # small rank to keep the test fast
    cache_path = tmp_path / "isdf_cache.h5"

    # Persist kernels for params_a
    isdf_cached = xtc.ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank, save_path=str(cache_path))
    isdf_cached = isdf_cached.isdf(params_a, save_path=str(cache_path), batch_size=256)

    # Load the cached file with different Jastrow parameters – should trigger recomputation
    isdf_loaded = xtc.ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank, save_path=str(cache_path))
    isdf_loaded = isdf_loaded.isdf(params_b, save_path=str(cache_path), batch_size=256)
    cc_cached = xtc_ccsd.RCCSD(mf, isdf_loaded, params_b, on_the_fly_vvvv=True)
    e_cached, _, _ = cc_cached.kernel()

    # Fresh computation with params_b
    isdf_fresh = xtc.ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank, save_path=None)
    isdf_fresh = isdf_fresh.isdf(params_b, batch_size=256)
    cc_fresh = xtc_ccsd.RCCSD(mf, isdf_fresh, params_b, on_the_fly_vvvv=True)
    e_fresh, _, _ = cc_fresh.kernel()

    assert jnp.allclose(e_cached, e_fresh, atol=1e-8)
