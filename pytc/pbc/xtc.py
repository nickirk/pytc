"""Gamma-point periodic xTC construction helpers."""

import jax.numpy as jnp

from pytc.integrals.xtc import ISDFXTC, XTC

from .tc import create_tc


def create_xtc(mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
    """Build periodic xTC corrections using the direct TC grid oracle."""
    tc = create_tc(
        mf,
        jastrow_factor=jastrow_factor,
        mo_coeff=mo_coeff,
        grid_lvl=grid_lvl,
    )
    return XTC(
        grid_points=tc.grid_points,
        weights=tc.weights,
        phi=tc.phi,
        grad_phi=tc.grad_phi,
        n_orb=tc.n_orb,
        grid_lvl=tc.grid_lvl,
        jastrow_factor=tc.jastrow_factor,
        mo_coeff=tc.mo_coeff,
        nocc=tc.nocc,
        mo_occ=jnp.asarray(mf.mo_occ),
        energy_nuc=float(mf.energy_nuc()),
    )


def create_xtc_fft(mf, jastrow_factor, mo_coeff=None, mesh=None):
    """Build a uniform-grid Gamma xTC object with FFT-backed integrals."""
    from .fft_tc import create_xtc_fft as _create_xtc_fft

    return _create_xtc_fft(
        mf, jastrow_factor, mo_coeff=mo_coeff, mesh=mesh
    )


def create_isdf_xtc_fft(
    mf,
    jastrow_factor,
    jastrow_params,
    mo_coeff=None,
    mesh=None,
    *,
    n_rank=None,
    is_incore=False,
    save_path=None,
    ls_grid_batch_size=16384,
    fixed_pivots=None,
    channel_factorized=True,
    kernel_kwargs=None,
):
    """Build the Gamma periodic ISDF xTC factors consumed by RCCSD.

    The orbital/gradient decomposition uses one shared pivot set.  ``U1``,
    ``U3``, and the ``L_aux`` input to the ``D/X`` mean-field-reduction
    algebra are built with the uniform-grid FFT pair backend.  The returned
    object therefore exposes the complete
    ``K1_kernel/K3_kernel/D/X`` contract required by
    :class:`pytc.solver.isdf_xtc_ccsd.RCCSD` without a direct grid-pair
    Jastrow construction in this helper.

    This is deliberately Gamma/real only, matching :func:`create_xtc_fft` and
    the current factor-direct solver.  ``kernel_kwargs`` is forwarded to
    :meth:`pytc.integrals.xtc.ISDFXTC.compute_delta_u_kernels` for production
    blocking controls.
    """
    fft_xtc = create_xtc_fft(
        mf, jastrow_factor, mo_coeff=mo_coeff, mesh=mesh
    )
    isdf_xtc = ISDFXTC.from_xtc(
        fft_xtc,
        n_rank=n_rank,
        is_incore=is_incore,
        save_path=save_path,
        ls_grid_batch_size=ls_grid_batch_size,
        fixed_pivots=fixed_pivots,
    )
    from .fft_tc import calc_isdf_kernels_fft, calc_isdf_l_aux_fft
    from pytc.utils.reuse import ReuseScope

    out_path = isdf_xtc.save_path
    xi_phi = isdf_xtc.xi_phi
    xi_grad = isdf_xtc.xi_grad
    if xi_phi is None or xi_grad is None:
        if out_path is None:
            raise RuntimeError("out-of-core ISDF decomposition has no backing path")
        import h5py

        with h5py.File(out_path, "r") as handle:
            xi_phi = handle["xi_phi"][:]
            xi_grad = handle["xi_grad"][:]

    reuse = ReuseScope(max_entries=1)
    u1, u3 = calc_isdf_kernels_fft(
        xi_phi,
        xi_grad,
        isdf_xtc.weights,
        isdf_xtc.grid_points,
        fft_xtc.fft_mesh,
        jastrow_factor,
        jastrow_params,
        channel_factorized=channel_factorized,
        _reuse=reuse,
    )
    l_aux = calc_isdf_l_aux_fft(
        xi_phi,
        isdf_xtc.weights,
        isdf_xtc.grid_points,
        fft_xtc.fft_mesh,
        jastrow_factor,
        jastrow_params,
        channel_factorized=channel_factorized,
        _reuse=reuse,
    )

    if out_path is not None:
        import h5py
        import numpy as np

        with h5py.File(out_path, "a") as handle:
            for name, value in (
                ("phi_isdf", isdf_xtc.phi_isdf),
                ("grad_phi_isdf", isdf_xtc.grad_phi_isdf),
                ("pivots", isdf_xtc.pivots),
                ("K1_kernel", u1),
                ("K3_kernel", u3),
            ):
                if name in handle:
                    del handle[name]
                handle.create_dataset(name, data=np.asarray(value))

    kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)
    if "save_path" in kwargs or "L_aux" in kwargs:
        raise ValueError("kernel_kwargs must not override save_path or L_aux")
    delta_u = isdf_xtc.compute_delta_u_kernels(
        jastrow_params,
        L_aux=l_aux,
        save_path=out_path,
        **kwargs,
    )
    kernels = {"K1_kernel": u1, "K3_kernel": u3, **delta_u}
    return isdf_xtc.replace(isdf_kernels=kernels, save_path=out_path)
