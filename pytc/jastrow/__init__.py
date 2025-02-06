"""Jastrow factor implementations."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Jastrow(ABC):
    """Base class for Jastrow factors with built-in parallel gradient computation."""

    def __init__(self, params, mol=None):
        """Initialize the Jastrow factor with parameters.
        
        Args:
            mf: Mean-field object (e.g., pyscf SCF object)
            mo_coeff: Molecular orbital coefficients
            params: Dictionary of Jastrow parameters (or any other format, as long as
                the __call__ and _process_grad_batch methods can interpret it)
            mol: Molecule object (e.g., pyscf Mole object)
        """
        self.mol = mol
        self.params = params

    @abstractmethod
    def __call__(self, r1, r2, r_nuc=None):
        """Evaluate Jastrow factor at given positions.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            r_nuc: Optional array of shape (N_atoms, 3) for nuclear coordinates
        
        Returns:
            Array of shape (N1, N2) where N1, N2 are the batch dimensions of r1, r2
        """
        pass

    @abstractmethod
    def _process_grad_batch(self, r1_batch, r2):
        """Process a batch of r1 points for gradient computation.
        
        Args:
            r1_batch: Array of shape (batch_size, 3)
            r2: Array of shape (N2, 3)
            
        Returns:
            Array of shape (batch_size, N2, 3) containing gradients
        """
        pass
    
    def _get_batch_size(self, n_points):
        """Determine batch size based on available threads."""
        with ThreadPoolExecutor() as executor:
            n_workers = executor._max_workers
        return max(1, n_points // (4 * n_workers))
    
    def grad(self, r1, r2=None, r_nuc=None):
        """Compute gradient using parallel processing over batches."""
        r2 = r2 if r2 is not None else r1
        r1 = np.atleast_2d(r1)
        r2 = np.atleast_2d(r2)
        
        n_points = len(r2)
        batch_size = self._get_batch_size(n_points)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, n_points, batch_size):
                batch = r1[i:i+batch_size]
                futures.append(
                    executor.submit(self._process_grad_batch, batch, r2)
                )
            results = [f.result() for f in futures]
        
        return np.concatenate(results, axis=0)
    
# Import concrete implementations
from .simple import SimpleJastrow
from .sm7 import SM7
from .sm17 import SM17
from .casino import CASINO

# Make classes available at package level
__all__ = ['SimpleJastrow', 'SM7', 'SM17', 'CASINO']