import numpy as np
from functools import partial
import jax
import jax.numpy as jnp

from pyscf.dft import numint
from pytc.lib import np_helper

einsum = partial(np.einsum, optimize=True)


def _detect_moved_electrons(walker):
    """Detect which electrons moved in each walker.
    
    Args:
        walker: Walker dataclass with move_mask
        
    Returns:
        List of tuples (walker_idx, electron_idx) for moved electrons
    """
    # Convert to numpy for easier indexing
    move_mask_np = np.array(walker.move_mask)
    # Vectorized detection using np.where
    walker_indices, electron_indices = np.where(move_mask_np)
    moved_positions = list(zip(walker_indices, electron_indices))
    
    return moved_positions


def _batch_eval_ao_for_moved(det, walker, moved_indices):
    """Evaluate AOs only for moved electron positions.
    
    Args:
        det: SlaterDet object
        walker: Walker dataclass
        moved_indices: List of (walker_idx, electron_idx) tuples
        
    Returns:
        Array of AO values for moved electrons, shape (n_moved, n_aos)
    """
    if len(moved_indices) == 0:
        # No electrons moved, return empty array
        return np.array([])
    
    # Extract positions of moved electrons using numpy indexing
    positions_np = np.array(walker.positions)
    moved_indices_arr = np.array(moved_indices)  # shape: (n_moved, 2)
    walker_indices = moved_indices_arr[:, 0]
    electron_indices = moved_indices_arr[:, 1]
    
    # Use advanced indexing to extract moved coordinates
    moved_coords = positions_np[walker_indices, electron_indices]  # shape: (n_moved, 3)
    
    # Batch evaluate AOs for all moved electrons at once
    ao_vals = numint.eval_ao(det.mol, moved_coords, deriv=0)
    
    return ao_vals


def _batch_eval_ao_grad_lap_for_moved(det, walker, moved_indices):
    """Evaluate AO values, gradients, and laplacians for moved electrons.
    
    Args:
        det: SlaterDet object
        walker: Walker dataclass
        moved_indices: List of (walker_idx, electron_idx) tuples
        
    Returns:
        Tuple of (ao_vals, ao_grad_vals, ao_lap_vals)
        - ao_vals: shape (n_moved, n_aos)
        - ao_grad_vals: shape (n_moved, n_aos, 3)
        - ao_lap_vals: shape (n_moved, n_aos)
    """
    if len(moved_indices) == 0:
        # No moves, return empty arrays
        n_aos = det.mol.nao_nr()
        return (np.zeros((0, n_aos)), 
                np.zeros((0, n_aos, 3)), 
                np.zeros((0, n_aos)))
    
    # Extract positions for moved electrons
    moved_indices_arr = np.array(moved_indices)
    walker_indices = moved_indices_arr[:, 0]
    electron_indices = moved_indices_arr[:, 1]
    
    # Get positions using advanced indexing (vectorized)
    moved_coords = np.array(walker.positions)[walker_indices, electron_indices, :]
    
    # Evaluate AO values with deriv=2 (includes values, 1st derivatives, 2nd derivatives)
    ao_result = numint.eval_ao(det.mol, moved_coords, deriv=2)
    
    # PySCF returns: [values, dx, dy, dz, dxx, dyy, dzz, dxy, dxz, dyz]
    # We need: values, [dx, dy, dz], and laplacian = dxx + dyy + dzz
    ao_vals = ao_result[0]           # shape: (n_moved, n_aos)
    ao_grad = ao_result[1:4]         # shape: (3, n_moved, n_aos)
    ao_lap = ao_result[[4,7,9]].sum(axis=0)  # sum dxx, dyy, dzz -> shape: (n_moved, n_aos)
    
    # Transpose gradient to (n_moved, n_aos, 3)
    ao_grad_vals = np.transpose(ao_grad, (1, 2, 0))
    
    return ao_vals, ao_grad_vals, ao_lap


def _update_slater_rows(walker, ao_vals, moved_indices, 
                        mo_coeff_alpha_occ, mo_coeff_beta_occ, n_alpha):
    """Update Slater matrix rows for moved electrons.
    
    Args:
        walker: Walker dataclass
        ao_vals: AO values for moved electrons, shape (n_moved, n_aos)
        moved_indices: List of (walker_idx, electron_idx) tuples
        mo_coeff_alpha_occ: MO coefficients for alpha electrons
        mo_coeff_beta_occ: MO coefficients for beta electrons
        n_alpha: Number of alpha electrons
        
    Returns:
        Tuple of (updated_slater_up, updated_slater_down)
    """
    if len(moved_indices) == 0:
        # No moves, return existing matrices
        return np.array(walker.slater_up), np.array(walker.slater_down)
    
    # Copy existing matrices to numpy arrays for updating
    slater_up = np.array(walker.slater_up)
    slater_down = np.array(walker.slater_down)
    
    # Convert moved_indices to arrays for vectorized operations
    moved_indices_arr = np.array(moved_indices)  # shape: (n_moved, 2)
    walker_indices = moved_indices_arr[:, 0]
    electron_indices = moved_indices_arr[:, 1]
    
    # Separate alpha and beta electrons
    alpha_mask = electron_indices < n_alpha
    beta_mask = ~alpha_mask
    
    # Update alpha electrons (vectorized)
    if np.any(alpha_mask):
        alpha_ao_vals = ao_vals[alpha_mask]  # shape: (n_alpha_moved, n_aos)
        alpha_mo_vals = alpha_ao_vals @ mo_coeff_alpha_occ  # shape: (n_alpha_moved, n_alpha)
        alpha_walker_idx = walker_indices[alpha_mask]
        alpha_electron_idx = electron_indices[alpha_mask]
        
        # Vectorized assignment using advanced indexing
        slater_up[alpha_walker_idx, alpha_electron_idx, :] = alpha_mo_vals
    
    # Update beta electrons (vectorized)
    if np.any(beta_mask):
        beta_ao_vals = ao_vals[beta_mask]  # shape: (n_beta_moved, n_aos)
        beta_mo_vals = beta_ao_vals @ mo_coeff_beta_occ  # shape: (n_beta_moved, n_beta)
        beta_walker_idx = walker_indices[beta_mask]
        beta_electron_idx = electron_indices[beta_mask] - n_alpha  # Convert to beta indexing
        
        # Vectorized assignment using advanced indexing
        slater_down[beta_walker_idx, beta_electron_idx, :] = beta_mo_vals
    
    return slater_up, slater_down


def _update_grad_lap_rows(walker, ao_grad_vals, ao_lap_vals, moved_indices, 
                          mo_coeff_alpha_occ, mo_coeff_beta_occ, n_alpha):
    """Update gradient and laplacian matrix rows for moved electrons.
    
    Args:
        walker: Walker dataclass
        ao_grad_vals: AO gradient values, shape (n_moved, n_aos, 3)
        ao_lap_vals: AO laplacian values, shape (n_moved, n_aos)
        moved_indices: List of (walker_idx, electron_idx) tuples
        mo_coeff_alpha_occ: MO coefficients for alpha electrons
        mo_coeff_beta_occ: MO coefficients for beta electrons
        n_alpha: Number of alpha electrons
        
    Returns:
        Tuple of (updated_grad_up, updated_grad_down, updated_lap_up, updated_lap_down)
    """
    if len(moved_indices) == 0:
        # No moves, return existing matrices
        return (np.array(walker.grad_up), np.array(walker.grad_down),
                np.array(walker.lap_up), np.array(walker.lap_down))
    
    # Copy existing matrices
    grad_up = np.array(walker.grad_up)
    grad_down = np.array(walker.grad_down)
    lap_up = np.array(walker.lap_up)
    lap_down = np.array(walker.lap_down)
    
    # Convert moved_indices to arrays
    moved_indices_arr = np.array(moved_indices)
    walker_indices = moved_indices_arr[:, 0]
    electron_indices = moved_indices_arr[:, 1]
    
    # Separate alpha and beta electrons
    alpha_mask = electron_indices < n_alpha
    beta_mask = ~alpha_mask
    
    # Update alpha electrons
    if np.any(alpha_mask):
        alpha_ao_grad = ao_grad_vals[alpha_mask]  # shape: (n_alpha_moved, n_aos, 3)
        alpha_ao_lap = ao_lap_vals[alpha_mask]    # shape: (n_alpha_moved, n_aos)
        alpha_walker_idx = walker_indices[alpha_mask]
        alpha_electron_idx = electron_indices[alpha_mask]
        
        # Transform gradients: (n_moved, n_aos, 3) @ (n_aos, n_alpha) -> (n_moved, n_alpha, 3)
        alpha_mo_grad = np.einsum('ijk,jl->ilk', alpha_ao_grad, mo_coeff_alpha_occ)
        grad_up[alpha_walker_idx, alpha_electron_idx, :, :] = alpha_mo_grad
        
        # Transform laplacians
        alpha_mo_lap = alpha_ao_lap @ mo_coeff_alpha_occ
        lap_up[alpha_walker_idx, alpha_electron_idx, :] = alpha_mo_lap
    
    # Update beta electrons
    if np.any(beta_mask):
        beta_ao_grad = ao_grad_vals[beta_mask]
        beta_ao_lap = ao_lap_vals[beta_mask]
        beta_walker_idx = walker_indices[beta_mask]
        beta_electron_idx = electron_indices[beta_mask] - n_alpha
        
        # Transform gradients
        beta_mo_grad = np.einsum('ijk,jl->ilk', beta_ao_grad, mo_coeff_beta_occ)
        grad_down[beta_walker_idx, beta_electron_idx, :, :] = beta_mo_grad
        
        # Transform laplacians
        beta_mo_lap = beta_ao_lap @ mo_coeff_beta_occ
        lap_down[beta_walker_idx, beta_electron_idx, :] = beta_mo_lap
    
    return grad_up, grad_down, lap_up, lap_down


# to be deprecated
@partial(jax.jit, static_argnums=(0,))
def value(det, walker):
    """JAX-compatible wrapper for SlaterDet.value()
    
    Args:
        det: SlaterDet object (static)
        walker: Walker dataclass with positions, move_mask, and cached matrices
        
    Returns:
        Tuple of (det_values, updated_walker) where:
            det_values: JAX array of shape (n_walkers,) 
            updated_walker: Walker with updated slater matrices, dets, and inverses
    """
    def value_callback(positions_np, move_mask_np, slater_up_np, slater_down_np, 
                       inv_up_np, inv_down_np, det_up_np, det_down_np):
        """Callback that calls SlaterDet.value with numpy arrays."""
        # Create a temporary numpy-based walker structure
        class NumpyWalker:
            def __init__(self):
                self.positions = positions_np
                self.move_mask = move_mask_np
                self.slater_up = slater_up_np
                self.slater_down = slater_down_np
                self.inv_up = inv_up_np
                self.inv_down = inv_down_np
                self.det_up = det_up_np
                self.det_down = det_down_np
        
        np_walker = NumpyWalker()
        det_values, updated_matrices = det.value(np_walker)
        
        # Return all updated values
        return (np.asarray(det_values),
                np.asarray(updated_matrices['slater_up']),
                np.asarray(updated_matrices['slater_down']),
                np.asarray(updated_matrices['inv_up']),
                np.asarray(updated_matrices['inv_down']),
                np.asarray(updated_matrices['det_up']),
                np.asarray(updated_matrices['det_down']))
    
    n_walkers = walker.positions.shape[0]
    n_alpha, n_beta = det.n_alpha, det.n_beta
    
    # Define output shapes
    det_values_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    slater_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha), jnp.float64)
    slater_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta), jnp.float64)
    inv_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha), jnp.float64)
    inv_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta), jnp.float64)
    det_up_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    det_down_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    
    det_values, slater_up, slater_down, inv_up, inv_down, det_up, det_down = jax.pure_callback(
        value_callback,
        (det_values_shape, slater_up_shape, slater_down_shape, 
         inv_up_shape, inv_down_shape, det_up_shape, det_down_shape),
        walker.positions, walker.move_mask, walker.slater_up, walker.slater_down,
        walker.inv_up, walker.inv_down, walker.det_up, walker.det_down
    )
    
    # Create updated walker
    updated_walker = walker.replace(
        slater_up=slater_up,
        slater_down=slater_down,
        inv_up=inv_up,
        inv_down=inv_down,
        det_up=det_up,
        det_down=det_down
    )
    
    return det_values, updated_walker

@partial(jax.jit, static_argnums=(0,))
def value_and_grad(det, walker):
    """JAX-compatible wrapper for SlaterDet.value_and_grad()
    
    Computes determinant value, gradients, and laplacians in one call,
    with selective updates based on move_mask.
    
    Args:
        det: SlaterDet object (static)
        walker: Walker dataclass with positions, move_mask, and cached matrices
        
    Returns:
        Tuple of (det_values, updated_walker) where:
            det_values: JAX array of shape (n_walkers,)
            updated_walker: Walker with all quantities updated (Slater, grad, lap)
    """
    def value_and_grad_callback(positions_np, move_mask_np, slater_up_np, slater_down_np,
                                inv_up_np, inv_down_np, det_up_np, det_down_np,
                                grad_up_np, grad_down_np, lap_up_np, lap_down_np):
        """Callback that calls SlaterDet.value_and_grad with numpy arrays."""
        # Create a temporary numpy-based walker structure
        class NumpyWalker:
            def __init__(self):
                self.positions = positions_np
                self.move_mask = move_mask_np
                self.slater_up = slater_up_np
                self.slater_down = slater_down_np
                self.inv_up = inv_up_np
                self.inv_down = inv_down_np
                self.det_up = det_up_np
                self.det_down = det_down_np
                self.grad_up = grad_up_np
                self.grad_down = grad_down_np
                self.lap_up = lap_up_np
                self.lap_down = lap_down_np
        
        np_walker = NumpyWalker()
        det_values, updated_data = det.value_and_grad(np_walker)
        
        # Return all updated values
        return (np.asarray(det_values),
                np.asarray(updated_data['slater_up']),
                np.asarray(updated_data['slater_down']),
                np.asarray(updated_data['inv_up']),
                np.asarray(updated_data['inv_down']),
                np.asarray(updated_data['det_up']),
                np.asarray(updated_data['det_down']),
                np.asarray(updated_data['grad_up']),
                np.asarray(updated_data['grad_down']),
                np.asarray(updated_data['lap_up']),
                np.asarray(updated_data['lap_down']))
    
    n_walkers = walker.positions.shape[0]
    n_alpha, n_beta = det.n_alpha, det.n_beta
    
    # Define output shapes
    det_values_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    slater_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha), jnp.float64)
    slater_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta), jnp.float64)
    inv_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha), jnp.float64)
    inv_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta), jnp.float64)
    det_up_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    det_down_shape = jax.ShapeDtypeStruct((n_walkers,), jnp.float64)
    grad_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha, 3), jnp.float64)
    grad_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta, 3), jnp.float64)
    lap_up_shape = jax.ShapeDtypeStruct((n_walkers, n_alpha, n_alpha), jnp.float64)
    lap_down_shape = jax.ShapeDtypeStruct((n_walkers, n_beta, n_beta), jnp.float64)
    
    det_values, slater_up, slater_down, inv_up, inv_down, det_up, det_down, \
    grad_up, grad_down, lap_up, lap_down = jax.pure_callback(
        value_and_grad_callback,
        (det_values_shape, slater_up_shape, slater_down_shape,
         inv_up_shape, inv_down_shape, det_up_shape, det_down_shape,
         grad_up_shape, grad_down_shape, lap_up_shape, lap_down_shape),
        walker.positions, walker.move_mask, walker.slater_up, walker.slater_down,
        walker.inv_up, walker.inv_down, walker.det_up, walker.det_down,
        walker.grad_up, walker.grad_down, walker.lap_up, walker.lap_down
    )
    
    # Create updated walker with all quantities
    updated_walker = walker.replace(
        slater_up=slater_up,
        slater_down=slater_down,
        inv_up=inv_up,
        inv_down=inv_down,
        det_up=det_up,
        det_down=det_down,
        grad_up=grad_up,
        grad_down=grad_down,
        lap_up=lap_up,
        lap_down=lap_down
    )
    
    return det_values, updated_walker

@partial(jax.jit, static_argnums=(0,))
def grad(det, coords):
    """JAX-compatible wrapper for SlaterDet.grad()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        
    Returns:
        Tuple of JAX arrays for (grad_up, grad_down) with shapes:
        ((n_walkers, n_alpha, n_alpha, 3), (n_walkers, n_beta, n_beta, 3))
    """
    def grad_callback(coords_np):
        matrix_out, grad_out = det.grad(np.array(coords_np))
        return (np.asarray(matrix_out[0]), np.asarray(matrix_out[1]),
                np.asarray(grad_out[0]), np.asarray(grad_out[1]))
    
    matrix_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    matrix_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    grad_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha, 3), jnp.float64)
    grad_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta, 3), jnp.float64)
    
    return jax.pure_callback(grad_callback, 
                           (matrix_up_shape, matrix_down_shape, 
                            grad_up_shape, grad_down_shape), 
                           coords)

@partial(jax.jit, static_argnums=(0,))
def laplacian(det, walker):
    """JAX-compatible wrapper for SlaterDet.laplacian()
    
    Args:
        det: SlaterDet object (static)
        walker: Walker dataclass
        
    Returns:
        Tuple of ((matrix_up, matrix_down), (grad_up, grad_down), (lap_up, lap_down), updated_walker)
    """
    def laplacian_callback(positions, slater_up, slater_down, inv_up, inv_down, 
                          det_up, det_down, grad_up, grad_down, lap_up, lap_down, 
                          move_mask):
        # Reconstruct numpy-based walker for the callback
        walker_np = type('Walker', (), {
            'positions': np.array(positions),
            'slater_up': np.array(slater_up),
            'slater_down': np.array(slater_down),
            'inv_up': np.array(inv_up),
            'inv_down': np.array(inv_down),
            'det_up': np.array(det_up),
            'det_down': np.array(det_down),
            'grad_up': np.array(grad_up),
            'grad_down': np.array(grad_down),
            'lap_up': np.array(lap_up),
            'lap_down': np.array(lap_down),
            'move_mask': np.array(move_mask)
        })()
        
        matrix_out, grad_out, lap_out = det.laplacian(walker_np)
        
        # Return just the matrix_out, grad_out, lap_out
        return (np.asarray(matrix_out[0]), np.asarray(matrix_out[1]),
                np.asarray(grad_out[0]), np.asarray(grad_out[1]),
                np.asarray(lap_out[0]), np.asarray(lap_out[1]))
    
    n_walkers = walker.positions.shape[0]
    matrix_up_shape = jax.ShapeDtypeStruct((n_walkers, det.n_alpha, det.n_alpha), jnp.float64)
    matrix_down_shape = jax.ShapeDtypeStruct((n_walkers, det.n_beta, det.n_beta), jnp.float64)
    grad_up_shape = jax.ShapeDtypeStruct((n_walkers, det.n_alpha, det.n_alpha, 3), jnp.float64)
    grad_down_shape = jax.ShapeDtypeStruct((n_walkers, det.n_beta, det.n_beta, 3), jnp.float64)
    lap_up_shape = jax.ShapeDtypeStruct((n_walkers, det.n_alpha, det.n_alpha), jnp.float64)
    lap_down_shape = jax.ShapeDtypeStruct((n_walkers, det.n_beta, det.n_beta), jnp.float64)
    
    matrix_up, matrix_down, grad_up, grad_down, lap_up, lap_down = jax.pure_callback(
        laplacian_callback,
        (matrix_up_shape, matrix_down_shape,
         grad_up_shape, grad_down_shape,
         lap_up_shape, lap_down_shape),
        walker.positions, walker.slater_up, walker.slater_down,
        walker.inv_up, walker.inv_down, walker.det_up, walker.det_down,
        walker.grad_up, walker.grad_down, walker.lap_up, walker.lap_down,
        walker.move_mask
    )
    
    # Update walker with new grad/lap using the values from pure_callback
    updated_walker = walker.replace(
        grad_up=grad_up,
        grad_down=grad_down,
        lap_up=lap_up,
        lap_down=lap_down
    )
    
    return (matrix_up, matrix_down), (grad_up, grad_down), (lap_up, lap_down), updated_walker

@partial(jax.jit, static_argnums=(0,))
def matrix(det, coords):
    """JAX-compatible wrapper for SlaterDet.matrix()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        
    Returns:
        Tuple of JAX arrays for (slater_up, slater_down) with shapes:
        ((n_walkers, n_alpha, n_alpha), (n_walkers, n_beta, n_beta))
    """
    def matrix_callback(coords_np):
        # Call SlaterDet method directly - it handles batching
        slater_up, slater_down = det.matrix(np.array(coords_np))
        return (np.asarray(slater_up), np.asarray(slater_down))
    
    up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    
    return jax.pure_callback(matrix_callback, (up_shape, down_shape), coords)

class SlaterDet:
    def __init__(self, mol, mo_coeff=None, nelec=None, excitations=None):
        """
        Args:
        mol: A PySCF mol object (provides integrals, eval_gto, etc.)
        mo_coeff: Either a single np.ndarray of shape (nAOs, nMOs) for RHF,
        or a tuple/list [mo_coeff_alpha, mo_coeff_beta] each of
        shape (nAOs, nMOs) for UHF.
        nelec: Number of electrons as a tuple (n_alpha, n_beta).
        excitations: Tuple of (alpha_excitations, beta_excitations) where each is a tuple of
                    (from_indices, to_indices) specifying which orbitals to remove and add.
                    Example: (([0,1], [5,6]), ([], [])) means:
                    - For alpha: remove electrons from orbitals 0,1 and add to orbitals 5,6
                    - For beta: no excitations (regular HF reference)
        """
        self.mol = mol
        if nelec is None:
            self.n_alpha, self.n_beta = mol.nelec 
        else:
            self.n_alpha, self.n_beta = nelec
    
        # Detect if mo_coeff is restricted or unrestricted:
        if isinstance(mo_coeff, (list, tuple)):
            # mo_coeff[0] = alpha, mo_coeff[1] = beta
            self.mo_coeff_alpha = mo_coeff[0]
            self.mo_coeff_beta = mo_coeff[1]
            self.unrestricted = True
        else:
            # Single set of coefficients, treat as RHF
            self.mo_coeff_alpha = mo_coeff
            self.mo_coeff_beta = mo_coeff  # identical for spin up/down
            self.unrestricted = False
    
        # Default occupied orbitals (HF reference)
        self.alpha_occ = list(range(self.n_alpha)) 
        self.beta_occ = list(range(self.n_beta))  

        # Apply excitations if specified
        if excitations is not None:
            alpha_exc, beta_exc = excitations
            
            # Handle alpha excitations
            if alpha_exc and len(alpha_exc) == 2:
                from_idx, to_idx = alpha_exc
                # Validate excitation indices
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for alpha excitations")
                
                # Apply excitations
                for i, a in zip(from_idx, to_idx):
                    if i not in self.alpha_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied alpha orbital {i}")
                    if a in self.alpha_occ:
                        raise ValueError(f"Cannot add electron to already occupied alpha orbital {a}")
                    self.alpha_occ.remove(i)  
                    self.alpha_occ.append(a)  
                self.alpha_occ.sort()  # Keep indices sorted
                
            # Handle beta excitations
            if beta_exc and len(beta_exc) == 2:
                from_idx, to_idx = beta_exc
                # Validate excitation indices
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for beta excitations")
                
                # Apply excitations
                for i, a in zip(from_idx, to_idx):
                    if i not in self.beta_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied beta orbital {i}")
                    if a in self.beta_occ:
                        raise ValueError(f"Cannot add electron to already occupied beta orbital {a}")
                    self.beta_occ.remove(i)  
                    self.beta_occ.append(a)  
                self.beta_occ.sort()  # Keep indices sorted

        # Store the occupied MO coefficients
        self.mo_coeff_alpha_occ = self.mo_coeff_alpha[:, self.alpha_occ]
        self.mo_coeff_beta_occ = self.mo_coeff_beta[:, self.beta_occ]
    
        # Stored determinant values and coordinates
        self.det_up = None
        self.det_down = None
        self.last_positions = None

    def __call__(self, coords, params=None, move_mask=None):
        """
        Convenience method to call value on a set of coordinates.
    
        Args:
            coords: (n_up + n_down, 3) electron positions
        Returns:
            float: The product of alpha and beta determinants
        """
        return self.value(coords, move_mask=move_mask)
    
    @property
    def n_electrons(self):
        """Return the total number of electrons."""
        return self.n_alpha + self.n_beta

    def ao2mo(self, ao_vals, mo_coeff_alpha, mo_coeff_beta, n_walkers, n_electrons):
        """Transform AO values to MO basis for both spins.
        
        Args:
            ao_vals: Array of AO values
            mo_coeff_alpha/beta: MO coefficients for each spin
            n_walkers: Number of walkers
            n_electrons: Total number of electrons
            
        Returns:
            Tuple of (alpha_vals, beta_vals) in MO basis
        """
        if not self.unrestricted:
            mo_vals = np.dot(ao_vals, mo_coeff_alpha).reshape(n_walkers, n_electrons, -1)
            return (mo_vals[:, :self.n_alpha, :self.n_alpha], 
                    mo_vals[:, self.n_alpha:, :self.n_beta])
        else:
            ao_up = ao_vals.reshape(n_walkers, n_electrons, -1)[:, :self.n_alpha].reshape(-1, ao_vals.shape[-1])
            ao_down = ao_vals.reshape(n_walkers, n_electrons, -1)[:, self.n_alpha:].reshape(-1, ao_vals.shape[-1])
            return (np.dot(ao_up, mo_coeff_alpha).reshape(n_walkers, self.n_alpha, self.n_alpha),
                    np.dot(ao_down, mo_coeff_beta).reshape(n_walkers, self.n_beta, self.n_beta))

    def matrix(self, coords):
        """
        Build the alpha/beta Slater matrices for all electrons given coords.
    
        Args:
            coords: (n_up + n_down, 3) array of electron coordinates
                   or (n_walkers, n_up + n_down, 3) for batched evaluation
                   
        Returns:
            slater_up, slater_down: the Slater matrices for alpha and beta spins
                                  If input is batched, returns arrays of shape:
                                  (n_walkers, n_up, n_up), (n_walkers, n_down, n_down)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        flat_coords = coords_batch.reshape(-1, 3)
        ao_vals_all = numint.eval_ao(self.mol, flat_coords, deriv=0)
        
        # Direct computation without caching
        if not self.unrestricted:
            slater_batch = np.dot(ao_vals_all, self.mo_coeff_alpha_occ).reshape(n_walkers, n_electrons, -1)
            up = slater_batch[:,:self.n_alpha, :self.n_alpha]
            down = slater_batch[:, self.n_alpha:, :self.n_beta]
        else:
            ao_vals_all = ao_vals_all.reshape(n_walkers, n_electrons, -1)
            ao_up = ao_vals_all[:, :self.n_alpha].reshape(-1, ao_vals_all.shape[-1])
            ao_down = ao_vals_all[:, self.n_alpha:].reshape(-1, ao_vals_all.shape[-1])
            up = np.dot(ao_up, self.mo_coeff_alpha_occ).reshape(n_walkers, self.n_alpha, self.n_alpha)
            down = np.dot(ao_down, self.mo_coeff_beta_occ).reshape(n_walkers, self.n_beta, self.n_beta)
            
        return up, down

    def grad(self, coords):
        """Compute the gradient and matrix of the Slater determinant.

        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            tuple: ((matrix_up, matrix_down), (grad_up, grad_down))
                 If input is batched:
                 matrix_up/down: (n_walkers, n_up/down, n_up/down)
                 grad_up/down: (n_walkers, n_up/down, n_up/down, 3)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        flat_coords = coords_batch.reshape(-1, 3)
        ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=1)
        
        # Direct computation
        matrix_up, matrix_down = self.ao2mo(ao_vals_deriv[0],
                                          self.mo_coeff_alpha_occ,
                                          self.mo_coeff_beta_occ,
                                          n_walkers, n_electrons)
        
        grad_up = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
        grad_down = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
        
        ao_grads = ao_vals_deriv[1:].transpose(1, 2, 0)
        for d in range(3):
            gup, gdown = self.ao2mo(ao_grads[..., d],
                                  self.mo_coeff_alpha_occ,
                                  self.mo_coeff_beta_occ,
                                  n_walkers, n_electrons)
            grad_up[..., d] = gup
            grad_down[..., d] = gdown
            
        return (matrix_up, matrix_down), (grad_up, grad_down)

    def laplacian(self, walker):
        """Compute laplacian, gradient and matrix of the Slater determinant using Walker.
                
        Args:
            walker: Walker dataclass (or numpy-based walker with same attributes)
                  
        Returns:
            tuple: ((matrix_up, matrix_down), 
                    (grad_up, grad_down),
                    (lap_up, lap_down),
                    updated_grad_lap_dict)
            where updated_grad_lap_dict contains keys 'grad_up', 'grad_down', 'lap_up', 'lap_down'
        """
        # Check if grad/lap matrices are uninitialized (all zeros)
        grad_up_np = np.array(walker.grad_up)
        matrices_uninitialized = np.allclose(grad_up_np, 0.0)
        
        #if matrices_uninitialized:
        if matrices_uninitialized:
            # Full recomputation needed - compute all from scratch
            coords_np = np.array(walker.positions)
            coords_batch, is_single = self._ensure_batch(coords_np)
            n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
            
            # Evaluate AOs with derivatives
            flat_coords = coords_batch.reshape(-1, 3)
            ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=2)
            
            # Get matrices
            matrix_up, matrix_down = self.ao2mo(ao_vals_deriv[0],
                                              self.mo_coeff_alpha_occ,
                                              self.mo_coeff_beta_occ,
                                              n_walkers, n_electrons)
            
            # Get gradients
            ao_grads = ao_vals_deriv[1:4].transpose(1, 2, 0)
            grad_up = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
            grad_down = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
            
            for d in range(3):
                gup_d, gdown_d = self.ao2mo(ao_grads[..., d],
                                      self.mo_coeff_alpha_occ,
                                      self.mo_coeff_beta_occ,
                                      n_walkers, n_electrons)
                grad_up[..., d] = gup_d
                grad_down[..., d] = gdown_d
                
            # Get laplacians
            ao_lapls = ao_vals_deriv[[4, 7, 9]].sum(axis=0)
            lap_up, lap_down = self.ao2mo(ao_lapls,
                                        self.mo_coeff_alpha_occ,
                                        self.mo_coeff_beta_occ,
                                        n_walkers, n_electrons)
            
            
            return (matrix_up, matrix_down), (grad_up, grad_down), (lap_up, lap_down)
        
        # Detect which electrons moved based on move_mask
        moved_indices = _detect_moved_electrons(walker)
        
        # Batch evaluate AOs with derivatives only for moved electrons
        ao_vals, ao_grad_vals, ao_lap_vals = _batch_eval_ao_grad_lap_for_moved(self, walker, moved_indices)
        
        # Update gradient and laplacian matrices for moved electrons
        updated_grad_up, updated_grad_down, updated_lap_up, updated_lap_down = _update_grad_lap_rows(
            walker, ao_grad_vals, ao_lap_vals, moved_indices,
            self.mo_coeff_alpha_occ, self.mo_coeff_beta_occ,
            self.n_alpha
        )
        
        # Use existing Slater matrices from walker (already updated by value())
        matrix_up = np.array(walker.slater_up)
        matrix_down = np.array(walker.slater_down)
        
        
        return (matrix_up, matrix_down), (updated_grad_up, updated_grad_down), (updated_lap_up, updated_lap_down)


    # to be deprecated
    def value(self, walker):
        """Compute determinant value using Walker dataclass.
        
        Args:
            walker: Walker dataclass (or numpy-based walker with same attributes)
            
        Returns:
            Tuple of (det_values, updated_matrices_dict) where:
                det_values: array of shape (n_walkers,)
                updated_matrices_dict: dict with keys 'slater_up', 'slater_down', 
                                      'inv_up', 'inv_down', 'det_up', 'det_down'
        """
        # Check if walker matrices are uninitialized (all zeros)
        # If det_up is all zeros, we need full recomputation
        det_up_np = np.array(walker.det_up)
        matrices_uninitialized = np.allclose(det_up_np, 0.0)
        
        if matrices_uninitialized:
            # Full recomputation needed - compute all matrices from scratch
            coords_np = np.array(walker.positions)
            slater_up, slater_down = self.matrix(coords_np)
            
            # Compute determinants
            det_up = np_helper.batched_det(slater_up)
            det_down = np_helper.batched_det(slater_down)
            
            # Compute inverses
            inv_up = np.linalg.inv(slater_up)
            inv_down = np.linalg.inv(slater_down)
            
            # Compute final determinant values
            det_values = det_up * det_down
            
            updated_matrices = {
                'slater_up': slater_up,
                'slater_down': slater_down,
                'inv_up': inv_up,
                'inv_down': inv_down,
                'det_up': det_up,
                'det_down': det_down
            }
            
            return det_values, updated_matrices
        
        # Detect which electrons moved based on move_mask
        moved_indices = _detect_moved_electrons(walker)
        
        # Batch evaluate AOs only for moved electrons
        ao_vals = _batch_eval_ao_for_moved(self, walker, moved_indices)
        
        # Update Slater matrix rows for moved electrons
        updated_slater_up, updated_slater_down = _update_slater_rows(
            walker, ao_vals, moved_indices, 
            self.mo_coeff_alpha_occ, self.mo_coeff_beta_occ, 
            self.n_alpha
        )
        
        # Determine which walkers had moves
        move_mask_np = np.array(walker.move_mask)
        walkers_with_moves = np.any(move_mask_np, axis=1)  # shape: (n_walkers,) boolean array
        
        # Compute determinants and inverses only for walkers with moves
        det_up = np.array(walker.det_up).copy()
        det_down = np.array(walker.det_down).copy()
        inv_up = np.array(walker.inv_up).copy()
        inv_down = np.array(walker.inv_down).copy()
        
        # Vectorized computation using np.where and batched operations
        # Extract matrices for walkers that moved
        if np.any(walkers_with_moves):
            # Get indices of walkers that moved
            moved_walker_indices = np.where(walkers_with_moves)[0]
            
            # Extract matrices for moved walkers
            slater_up_moved = updated_slater_up[moved_walker_indices]
            slater_down_moved = updated_slater_down[moved_walker_indices]
            
            # Compute determinants using batched_det
            new_det_up = np_helper.batched_det(slater_up_moved)
            new_det_down = np_helper.batched_det(slater_down_moved)
            
            # Compute inverses using vectorized linalg.inv
            # np.linalg.inv can handle batched inputs
            new_inv_up = np.linalg.inv(slater_up_moved)
            new_inv_down = np.linalg.inv(slater_down_moved)
            
            # Update only the moved walkers
            det_up[moved_walker_indices] = new_det_up
            det_down[moved_walker_indices] = new_det_down
            inv_up[moved_walker_indices] = new_inv_up
            inv_down[moved_walker_indices] = new_inv_down
        
        # Compute final determinant values
        det_values = det_up * det_down
        
        # Return det values and updated matrices
        updated_matrices = {
            'slater_up': updated_slater_up,
            'slater_down': updated_slater_down,
            'inv_up': inv_up,
            'inv_down': inv_down,
            'det_up': det_up,
            'det_down': det_down
        }
        
        return det_values, updated_matrices

    def value_and_grad(self, walker):
        """Compute determinant value, gradients, and laplacians using Walker dataclass.
        
        This method updates Slater matrices, gradients, and laplacians for electrons
        where move_mask=True, using selective updates for efficiency.
        
        Args:
            walker: Walker dataclass with positions, move_mask, and cached matrices
            
        Returns:
            Tuple of (det_values, updated_walker_data) where:
                det_values: array of shape (n_walkers,)
                updated_walker_data: dict with keys 'slater_up', 'slater_down',
                                    'inv_up', 'inv_down', 'det_up', 'det_down',
                                    'grad_up', 'grad_down', 'lap_up', 'lap_down'
        """
        # Check if walker matrices are uninitialized (all zeros)
        det_up_np = np.array(walker.det_up)
        matrices_uninitialized = np.allclose(det_up_np, 0.0)
        
        if matrices_uninitialized:
            # Full recomputation needed - compute everything from scratch
            coords_np = np.array(walker.positions)
            coords_batch, is_single = self._ensure_batch(coords_np)
            n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
            
            # Evaluate AOs with derivatives
            flat_coords = coords_batch.reshape(-1, 3)
            ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=2)
            
            # Get Slater matrices
            slater_up, slater_down = self.ao2mo(ao_vals_deriv[0],
                                              self.mo_coeff_alpha_occ,
                                              self.mo_coeff_beta_occ,
                                              n_walkers, n_electrons)
            
            # Compute determinants
            det_up = np_helper.batched_det(slater_up)
            det_down = np_helper.batched_det(slater_down)
            det_values = det_up * det_down
            
            # Compute inverses
            inv_up = np.linalg.inv(slater_up)
            inv_down = np.linalg.inv(slater_down)
            
            # Get gradients
            ao_grads = ao_vals_deriv[1:4].transpose(1, 2, 0)
            grad_up = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
            grad_down = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
            
            for d in range(3):
                gup_d, gdown_d = self.ao2mo(ao_grads[..., d],
                                      self.mo_coeff_alpha_occ,
                                      self.mo_coeff_beta_occ,
                                      n_walkers, n_electrons)
                grad_up[..., d] = gup_d
                grad_down[..., d] = gdown_d
                
            # Get laplacians
            ao_lapls = ao_vals_deriv[[4, 7, 9]].sum(axis=0)
            lap_up, lap_down = self.ao2mo(ao_lapls,
                                        self.mo_coeff_alpha_occ,
                                        self.mo_coeff_beta_occ,
                                        n_walkers, n_electrons)
            
            updated_data = {
                'slater_up': slater_up,
                'slater_down': slater_down,
                'inv_up': inv_up,
                'inv_down': inv_down,
                'det_up': det_up,
                'det_down': det_down,
                'grad_up': grad_up,
                'grad_down': grad_down,
                'lap_up': lap_up,
                'lap_down': lap_down
            }
            
            return det_values, updated_data
        
        # Selective update based on move_mask
        moved_indices = _detect_moved_electrons(walker)

        if len(moved_indices) == 0:
            # No moves, return existing values
            det_values = np.array(walker.det_up) * np.array(walker.det_down)
            updated_data = {
                'slater_up': np.array(walker.slater_up),
                'slater_down': np.array(walker.slater_down),
                'inv_up': np.array(walker.inv_up),
                'inv_down': np.array(walker.inv_down),
                'det_up': np.array(walker.det_up),
                'det_down': np.array(walker.det_down),
                'grad_up': np.array(walker.grad_up),
                'grad_down': np.array(walker.grad_down),
                'lap_up': np.array(walker.lap_up),
                'lap_down': np.array(walker.lap_down)
            }
            return det_values, updated_data
        
        # Batch evaluate AOs with derivatives for moved electrons
        ao_vals, ao_grad_vals, ao_lap_vals = _batch_eval_ao_grad_lap_for_moved(self, walker, moved_indices)
        
        # Update Slater matrix rows
        updated_slater_up, updated_slater_down = _update_slater_rows(
            walker, ao_vals, moved_indices,
            self.mo_coeff_alpha_occ, self.mo_coeff_beta_occ,
            self.n_alpha
        )
        
        # Update gradient and laplacian rows
        updated_grad_up, updated_grad_down, updated_lap_up, updated_lap_down = _update_grad_lap_rows(
            walker, ao_grad_vals, ao_lap_vals, moved_indices,
            self.mo_coeff_alpha_occ, self.mo_coeff_beta_occ,
            self.n_alpha
        )
        
        # Determine which walkers had moves
        move_mask_np = np.array(walker.move_mask)
        walkers_with_moves = np.any(move_mask_np, axis=1)
        
        # Compute determinants and inverses only for walkers with moves
        det_up = np.array(walker.det_up).copy()
        det_down = np.array(walker.det_down).copy()
        inv_up = np.array(walker.inv_up).copy()
        inv_down = np.array(walker.inv_down).copy()
        
        if np.any(walkers_with_moves):
            # TODO: use fast update formulas here instead of full recompute
            moved_walker_indices = np.where(walkers_with_moves)[0]
            
            # Extract matrices for moved walkers
            slater_up_moved = updated_slater_up[moved_walker_indices]
            slater_down_moved = updated_slater_down[moved_walker_indices]
            
            # Compute determinants
            new_det_up = np_helper.batched_det(slater_up_moved)
            new_det_down = np_helper.batched_det(slater_down_moved)
            
            # Compute inverses
            new_inv_up = np.linalg.inv(slater_up_moved)
            new_inv_down = np.linalg.inv(slater_down_moved)
            
            # Update only the moved walkers
            det_up[moved_walker_indices] = new_det_up
            det_down[moved_walker_indices] = new_det_down
            inv_up[moved_walker_indices] = new_inv_up
            inv_down[moved_walker_indices] = new_inv_down
        
        # Compute final determinant values
        det_values = det_up * det_down
        
        updated_data = {
            'slater_up': updated_slater_up,
            'slater_down': updated_slater_down,
            'inv_up': inv_up,
            'inv_down': inv_down,
            'det_up': det_up,
            'det_down': det_down,
            'grad_up': updated_grad_up,
            'grad_down': updated_grad_down,
            'lap_up': updated_lap_up,
            'lap_down': updated_lap_down
        }
        
        return det_values, updated_data

    def _ensure_batch(self, coords):
        """Ensure coords are in batched format.
    
        Args:
            coords: Array of shape (n_electrons, 3) or (n_walkers, n_electrons, 3)
        
        Returns:
            tuple: (batched_coords, is_single) where
                   batched_coords has shape (n_walkers, n_electrons, 3)
                   is_single is True if original input was single walker
        """
        if len(coords.shape) == 3:
            return coords, False
        else:
            return coords[np.newaxis, :, :], True