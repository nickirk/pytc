import jax
import jax.numpy as jnp

def make_slater_jastrow(hf_solution, jastrow_apply, ncusp_apply=None):
    """
    Creates a Slater-Jastrow wavefunction.
    
    Args:
        hf_solution: Object with eval_slater method (from ferminet.pretrain.get_hf)
        jastrow_apply: Function (r_ee, params, nspins, r_ae=None) -> val
        ncusp_apply: Optional function (r_ae, params, nspins) -> val
        
    Returns:
        log_psi: Function (params, pos, spins, atoms, charges) -> (sign, log_abs)
    """
    
    def log_psi(params, pos, spins, atoms, charges):
        """
        Evaluates the wavefunction.
        
        Args:
            params: Dictionary of parameters. 
                    Must contain 'jastrow' for jastrow_apply.
                    If ncusp_apply is provided, must contain 'ncusp'.
            pos: Electron positions (nelec*3)
            spins: Electron spins (nelec) - usually unused by simple jastrows but passed for interface
            atoms: Atom positions (natom, 3)
            charges: Atom charges (natom)
        """
        # Determine electrons tuple from hf_solution
        # hf_solution is scf.Scf object which wraps pyscf molecule
        if hasattr(hf_solution, 'nelectrons') and hf_solution.nelectrons is not None:
             electrons = hf_solution.nelectrons
        elif hasattr(hf_solution, '_mol'):
             electrons = hf_solution._mol.nelec
        else:
             raise ValueError("Cannot determine electrons tuple from hf_solution")
        
        # MCMC with atoms reshapes pos to (nelec, 1, 3). We need to flatten it to (nelec*3,)
        if pos.ndim > 1:
            pos = pos.reshape(-1)

        # HF part
        sign, log_det = hf_solution.eval_slater(pos, electrons)
        
        # Jastrow part
        pos_reshaped = pos.reshape((-1, 3))
        ee = pos_reshaped[None, :, :] - pos_reshaped[:, None, :]
        
        # Add identity to diagonal to avoid 0 in norm
        n = ee.shape[0]
        ee = ee + jnp.eye(n)[..., None]
        
        r_ee = jnp.linalg.norm(ee, axis=-1)
        # Mask diagonal
        r_ee = r_ee * (1.0 - jnp.eye(n))
        r_ee = r_ee[..., None] # (nelec, nelec, 1)
        
        # Compute r_ae (needed for ncusp and potentially BH)
        ae = pos_reshaped[:, None, :] - atoms[None, :, :]
        r_ae = jnp.linalg.norm(ae, axis=-1) # (nelec, natom)
        
        # Call jastrow_apply with r_ae
        # Note: jastrow_apply must accept r_ae as kwarg or *args
        if jastrow_apply is not None:
            j_val = jastrow_apply(r_ee, params.get('jastrow', {}), electrons, r_ae=r_ae)
        else:
            j_val = 0.0
        
        # Nuclear Cusp Jastrow
        if ncusp_apply is not None:
            j_val = j_val + ncusp_apply(r_ae, params.get('ncusp', {}), electrons)
        
        return sign, log_det + j_val

    return log_psi
