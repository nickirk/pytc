from ferminet import pretrain

def get_hf_det(molecule, nspins, basis='sto-3g', restricted=True):
    """
    Wrapper around ferminet.pretrain.get_hf to get Hartree-Fock solution.
    
    Args:
        molecule: List of system.Atom objects
        nspins: Tuple of (n_up, n_down)
        basis: Basis set name
        restricted: Whether to use restricted HF
        
    Returns:
        scf.Scf object with eval_slater method
    """
    return pretrain.get_hf(
        molecule=molecule,
        nspins=nspins,
        basis=basis,
        restricted=restricted
    )
