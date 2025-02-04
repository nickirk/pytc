"""This module implements different types of Jastrow factors."""

import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

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
        
        n_points = len(r1)
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

class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    
    def __call__(self, r1, r2):
        """Evaluate Jastrow factor at given positions."""
        r1 = np.atleast_2d(r1)  # Ensure 2D array with shape (N, 3)
        r2 = np.atleast_2d(r2)  # Ensure 2D array with shape (M, 3)
        
        delta_r = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        result = 0.5*1./self.params[0]*np.exp(-self.params[0] * np.linalg.norm(delta_r, axis=-1))
        
        # Handle single point inputs
        if r1.shape[0] == 1 and r2.shape[0] == 1:
            result = result.reshape(1, 1)
            
        return result
    
    def _process_grad_batch(self, r1_batch, r2):
        """Process a batch of r1 points efficiently using vectorized operations."""
        # Reshape for broadcasting
        r1_batch = np.asarray(r1_batch)[:, None, :]  # (batch, 1, 3)
        r2 = np.asarray(r2)[None, :, :]             # (1, N2, 3)
        
        # Vectorized operations for the batch
        diff = r1_batch - r2                         # (batch, N2, 3)
        norm = np.linalg.norm(diff, axis=-1)[..., None]  # (batch, N2, 1)
        norm = np.where(norm == 0, 1.0, norm)
        
        # Compute jastrow values efficiently for the batch
        r = norm.squeeze(-1)  # Remove last dimension for jastrow calculation
        jastrow_values = 0.5/self.params[0] * np.exp(-self.params[0] * r)[..., None]
        
        return -0.5*self.params[0] * diff / norm * jastrow_values

class SM7(Jastrow):
    """Schmidt-Moskowitz 7-parameter Jastrow factor."""
    
    # Class-level coefficient table
    _coeff_table = {
        'He': {(0,0,1): 0.50000, (0,0,2): 0.50516, (0,0,3): -0.19313, (0,0,4): 0.30276,
               (2,0,0): -0.16995, (3,0,0): -0.34505, (4,0,0): -0.54777},
        'Li': {(0,0,1): 0.50000, (0,0,2): 0.03104, (0,0,3): 0.48928, (0,0,4): -0.62908,
               (2,0,0): -0.07185, (3,0,0): -0.48761, (4,0,0): 0.40450},
        'Be': {(0,0,1): 0.50000, (0,0,2): -0.05254, (0,0,3): 0.15355, (0,0,4): -0.30549,
               (2,0,0): -0.11928, (3,0,0): -0.17144, (4,0,0): 0.16652},
        'B':  {(0,0,1): 0.50000, (0,0,2): -0.13852, (0,0,3): -0.06687, (0,0,4): -0.02026,
               (2,0,0): -0.12573, (3,0,0): -0.05320, (4,0,0): 0.06421},
        'C':  {(0,0,1): 0.50000, (0,0,2): -0.14368, (0,0,3): -0.34102, (0,0,4): 0.30267,
               (2,0,0): -0.12272, (3,0,0): -0.05622, (4,0,0): 0.08462},
        'N':  {(0,0,1): 0.50000, (0,0,2): -0.41390, (0,0,3): 0.10406, (0,0,4): 0.06374,
               (2,0,0): -0.12400, (3,0,0): 0.01909, (4,0,0): -0.00383},
        'O':  {(0,0,1): 0.50000, (0,0,2): -0.57077, (0,0,3): 0.44725, (0,0,4): -0.16075,
               (2,0,0): -0.11696, (3,0,0): -0.01442, (4,0,0): 0.03312},
        'F':  {(0,0,1): 0.50000, (0,0,2): -0.73946, (0,0,3): 0.81463, (0,0,4): -0.41861,
               (2,0,0): -0.11872, (3,0,0): -0.01973, (4,0,0): 0.02779},
        'Ne': {(0,0,1): 0.50000, (0,0,2): -0.79266, (0,0,3): 1.05232, (0,0,4): -0.65615,
               (2,0,0): -0.13312, (3,0,0): -0.00131, (4,0,0): 0.09083}
    }

    def __init__(self, params=None, atom=None):
        """Initialize with either explicit parameters or atom name."""
        if atom is not None:
            if atom not in self._coeff_table:
                raise ValueError(f"Parameters not available for atom: {atom}")
            params = self._coeff_table[atom]
        elif params is None:
            raise ValueError("Either params or atom must be provided")
        super().__init__(params)
    
    def _scaled_r(self, r):
        """Convert distance to scaled distance r/(1+r)."""
        return r / (1.0 + r)
    
    def _scaled_r_grad(self, r):
        """Gradient of scaled distance with respect to r."""
        return 1.0 / (1.0 + r)**2
    
    def __call__(self, r1, r2, r_nuc=None):
        """Evaluate SM7 Jastrow factor.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            r_nuc: Ignored (assumes nucleus at origin)
        """
        r1 = np.asarray(r1)[..., np.newaxis, :]
        r2 = np.asarray(r2)[np.newaxis, ...]
        
        # Get electron-nucleus distances
        r1_dist = np.sqrt(np.sum(r1 * r1, axis=-1))
        r2_dist = np.sqrt(np.sum(r2 * r2, axis=-1))
        
        # Get electron-electron distances
        diff = r1 - r2
        r12_dist = np.sqrt(np.sum(diff * diff, axis=-1))
        #r12_dist = np.where(r12_dist < 1e-10, 1e-10, r12_dist)
        
        # Convert to scaled distances
        r1_scaled = self._scaled_r(r1_dist)
        r2_scaled = self._scaled_r(r2_dist)
        r12_scaled = self._scaled_r(r12_dist)
        
        # Compute sum over m,n,o terms
        result = np.zeros_like(r12_dist)
        for (m,n,o), coeff in self.params.items():
            if m == n: 
                coeff_ = 0.5 * coeff
            term = coeff_ * (r1_scaled**m * r2_scaled**n + r2_scaled**m * r1_scaled**n) * r12_scaled**o
            result += term
            
        return result

    def _process_grad_batch(self, r1_batch, r2):
        """Process a batch of r1 points efficiently using vectorized operations."""
        r1_batch = np.asarray(r1_batch)
        r2 = np.asarray(r2)
        
        r1_expanded = r1_batch[:, np.newaxis, :]
        r2_expanded = r2[np.newaxis, :, :]
        
        r1_dist = np.sqrt(np.sum(r1_expanded * r1_expanded, axis=-1, keepdims=True))
        r1_grad_dir = r1_expanded / np.where(r1_dist < 1e-10, 1e-10, r1_dist)
        
        diff = r1_expanded - r2_expanded
        r12_dist = np.sqrt(np.sum(diff * diff, axis=-1, keepdims=True))
        r12_grad_dir = diff / np.where(r12_dist < 1e-10, 1e-10, r12_dist)
        
        r1_scaled = self._scaled_r(r1_dist)
        r2_scaled = self._scaled_r(np.sqrt(np.sum(r2_expanded * r2_expanded, axis=-1, keepdims=True)))
        r12_scaled = self._scaled_r(r12_dist)
        
        r1_scaled_grad = self._scaled_r_grad(r1_dist) * r1_grad_dir
        r12_scaled_grad = self._scaled_r_grad(r12_dist) * r12_grad_dir
        
        total_grad = np.zeros_like(diff)
        
        for (m,n,o), coeff in self.params.items():
            if m == n: 
                coeff_ = 0.5 * coeff
            else:
                coeff_ = coeff
            if m > 0:
                grad = coeff_ * m * r1_scaled**(m-1) * r2_scaled**n * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:
                grad = coeff_ * r1_scaled**m * r2_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
            
            if n > 0:
                grad = coeff_ * n * r1_scaled**(n-1) * r2_scaled**m * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:
                grad = coeff_ * r2_scaled**m * r1_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
        
        return total_grad

class SM17(SM7):
    """Schmidt-Moskowitz 17-parameter Jastrow factor."""
    
    # Class-level coefficient table
    _coeff_table = {
        'He': {(0,0,1): 0.50000, (0,0,2): 0.09239, (0,0,3): -0.38664, (0,0,4): 0.95731,
               (2,0,0): 0.23208, (3,0,0): -0.45032, (4,0,0): 0.82777, (2,2,0): -4.15388,
               (2,0,2): 0.80622, (2,2,2): 10.19704, (4,0,2): -4.96259, (2,0,4): -1.35647,
               (4,2,2): -5.90907, (6,0,2): 0.90343, (4,0,4): 5.50739, (2,2,4): -0.03154,
               (2,0,6): -1.05186},
        'Li': {(0,0,1): 0.50000, (0,0,2): -0.01380, (0,0,3): -0.45313, (0,0,4): 0.71032,
               (2,0,0): 0.30248, (3,0,0): -0.52643, (4,0,0): 1.93859, (2,2,0): -5.14421,
               (2,0,2): 1.38788, (2,2,2): 9.01474, (4,0,2): -9.05453, (2,0,4): 0.03344,
               (4,2,2): 3.11255, (6,0,2): 2.59001, (4,0,4): 4.60054, (2,2,4): -8.58519,
               (2,0,6): -0.92355},
        'Be': {(0,0,1): 0.50000, (0,0,2): -0.06478, (0,0,3): -0.64088, (0,0,4): 1.37713,
               (2,0,0): 0.38220, (3,0,0): -0.40248, (4,0,0): 1.38175, (2,2,0): -5.23497,
               (2,0,2): 0.65456, (2,2,2): 13.83365, (4,0,2): -5.39941, (2,0,4): -2.66066,
               (4,2,2): -2.13267, (6,0,2): -0.80566, (4,0,4): 8.42055, (2,2,4): -4.73764,
               (2,0,6): -1.49529},
        'B': {(0,0,1): 0.50000, (0,0,2): -0.18185, (0,0,3): -0.26288, (0,0,4): 0.77686,
              (2,0,0): 0.31118, (3,0,0): -0.14403, (4,0,0): 0.51106, (2,2,0): -4.17149,
              (2,0,2): -0.45334, (2,2,2): 13.71788, (4,0,2): -1.54596, (2,0,4): -1.35381,
              (4,2,2): -3.73788, (6,0,2): -1.54388, (4,0,4): 5.64542, (2,2,4): -4.74009,
              (2,0,6): -1.35761},
        'C': {(0,0,1): 0.50000, (0,0,2): -0.23934, (0,0,3): -0.51729, (0,0,4): 1.02146,
              (2,0,0): 0.21610, (3,0,0): -0.02774, (4,0,0): 0.28173, (2,2,0): -3.71952,
              (2,0,2): -0.28184, (2,2,2): 13.13613, (4,0,2): -1.02414, (2,0,4): -1.25233,
              (4,2,2): -4.62167, (6,0,2): -1.32220, (4,0,4): 5.22850, (2,2,4): -4.17394,
              (2,0,6): -1.64963},
        'N': {(0,0,1): 0.50000, (0,0,2): -0.37065, (0,0,3): -0.60669, (0,0,4): 1.17279,
              (2,0,0): 0.07317, (3,0,0): -0.05622, (4,0,0): 0.08462, (2,2,0): -3.71952,
              (2,0,2): -0.28184, (2,2,2): 13.13613, (4,0,2): -1.02414, (2,0,4): -1.25233,
              (4,2,2): -4.62167, (6,0,2): -1.32220, (4,0,4): 5.22850, (2,2,4): -4.17394,
              (2,0,6): -1.64963},
        'O': {(0,0,1): 0.50000, (0,0,2): -0.57077, (0,0,3): 0.44725, (0,0,4): -0.16075,
              (2,0,0): -0.11696, (3,0,0): -0.01442, (4,0,0): 0.03312, (2,2,0): -3.71952,
              (2,0,2): -0.28184, (2,2,2): 13.13613, (4,0,2): -1.02414, (2,0,4): -1.25233,
              (4,2,2): -4.62167, (6,0,2): -1.32220, (4,0,4): 5.22850, (2,2,4): -4.17394,
              (2,0,6): -1.64963},
        'F': {(0,0,1): 0.50000, (0,0,2): -0.73946, (0,0,3): 0.81463, (0,0,4): -0.41861,
              (2,0,0): -0.11872, (3,0,0): -0.01973, (4,0,0): 0.02779, (2,2,0): -3.71952,
              (2,0,2): -0.28184, (2,2,2): 13.13613, (4,0,2): -1.02414, (2,0,4): -1.25233,
              (4,2,2): -4.62167, (6,0,2): -1.32220, (4,0,4): 5.22850, (2,2,4): -4.17394,
              (2,0,6): -1.64963},
        'Ne': {(0,0,1): 0.50000, (0,0,2): -0.79266, (0,0,3): 1.05232, (0,0,4): -0.65615,
               (2,0,0): -0.13312, (3,0,0): -0.00131, (4,0,0): 0.09083, (2,2,0): -3.71952,
               (2,0,2): -0.28184, (2,2,2): 13.13613, (4,0,2): -1.02414, (2,0,4): -1.25233,
               (4,2,2): -4.62167, (6,0,2): -1.32220, (4,0,4): 5.22850, (2,2,4): -4.17394,
               (2,0,6): -1.64963}
    }

class CASINO(Jastrow):
    """Jastrow factor based on CASINO format parameters."""
    
    def _get_nuc_groups(self, is_symm=False):
        """Generate nucleus labels and groups based on atomic charges.
        
        Args:
            is_symm: Whether to use symmetry-based grouping

        Returns:
            nuc_groups: dict of nucleus label groups, e.g. {'n1': ['n1', 'n2'], 'n3': ['n3']}
            
        """
        if self.nuc_groups is not None:
            return self.nuc_groups, self.nuc_coords
        
        if self.mol is None:
            raise ValueError("mol object must be set to use nuclear coordinates")
            
        charges = self.mol.atom_charges()
        
        if is_symm:
            coords = self._get_nuc_coords()
            raise NotImplementedError("Symmetry-based grouping not implemented yet!")

        
        # Track unique charges we've seen
        charge_to_label = {}
        current_label_num = 1
        
        # First pass: assign labels
        nucleus_labels = []
        for charge in charges:
            label = f'n{current_label_num}'
            charge_to_label[charge] = label
            current_label_num += 1
            nucleus_labels.append(charge_to_label[charge])
            
        # Second pass: create groups
        nuc_groups = {}
        for label in nucleus_labels:
            # Extract numeric part from label (e.g. 'n1' -> 1)
            num = int(label[1:])
            # Find lowest numeric label in each group
            group_key = min(n for n in nucleus_labels if int(n[1:]) <= num)
            # Initialize group if needed
            if group_key not in nuc_groups:
                nuc_groups[group_key] = []
            # Add current label to appropriate group
            nuc_groups[group_key].append(label)
        
        self.nuc_groups = nuc_groups
            
        return self.nuc_groups 
    
    def _get_nuc_coords(self):
        """Get nuclear coordinates from molecule object."""
        if self.mol is None:
            raise ValueError("mol object must be set to use nuclear coordinates")
        self.nuc_coords =self.mol.atom_coords() 
        return self.nuc_coords
    
    def __call__(self, r1, r2, r_nuc=None, nuc_groups=None):
        """Evaluate CASINO Jastrow factor.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            r_nuc: Array of shape (N_atoms, 3) for nuclear coordinates
            nuc_groups: Dict of nucleus labels for grouping, e.g., {'n1': ['n1', 'n2'], 'n3': ['n3']}
        """
        if r_nuc is None:
            r_nuc = self._get_nuc_coords()
        
        if nuc_groups is None:
            nuc_groups = self._get_nuc_groups()

        
        total = 0.0
        for term in self.params:
            if term['rank'] == [2, 0]:
                total += self.compute_term_2e0n(term, r1, r2)
            elif term['rank'] == [1, 1]:
                total += self.compute_term_1e1n(term, r1, r_nuc, nuc_groups)
            elif term['rank'] == [2, 1]:
                total += self.compute_term_2e1n(term, r1, r2, r_nuc, nuc_groups)
            elif term['rank'] == [1, 2]:
                total += self.compute_term_1e2n(term, r1, r_nuc, nuc_groups)
        return total

    def compute_term_2e0n(self, term, r1, r2):
        """Compute electron-electron terms using vectorized operations."""
        # Check for spin-dependent terms
        if '1=2' not in term['rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
            
        # Get cutoff parameters
        L = term['e-e cutoff']['Parameters']['Channel 1-2']['L'][0]
        C = term['e-e cutoff']['Constants']['C']
        order = term['e-e basis']['order']
        
        # Compute all distances
        diff = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]  # shape: (n_grid, n_grid, 3)
        r_ij = np.sqrt(np.sum(diff * diff, axis=-1))  # shape: (n_grid, n_grid)
        
        # Create mask for cutoff
        mask = r_ij < L
        
        # Initialize result array
        result = np.zeros_like(r_ij)
        
        # Compute cutoff function where mask is True
        cutoff = np.where(mask, (1 - r_ij/L)**C, 0.0)
        
        # Get parameters for this channel
        channel_params = term['Linear parameters']['Channel 1-2']
        
        # Compute power series terms
        for nu in range(2, order + 1):
            param_key = f'c_{nu}'
            if param_key in channel_params:
                param = channel_params[param_key][0]  # Get parameter value
                # Only compute powers where mask is True
                power_term = np.where(mask, r_ij**nu, 0.0)
                result += param * power_term * cutoff
            
        return result

    def _compute_e_n_distances(self, r1, r_nuc):
        """Compute electron-nuclear distances efficiently.
        
        Args:
            r1: Array of shape (N_e, 3) for electron positions
            r_nuc: Array of shape (N_nuc, 3) for nuclear positions
            
        Returns:
            Array of shape (N_e, N_nuc) containing e-n distances
        """
        r1 = np.atleast_2d(r1)[:, np.newaxis, :]  # (N_e, 1, 3)
        r_nuc = np.atleast_2d(r_nuc)[np.newaxis, :, :]  # (1, N_nuc, 3)
        
        diff = r1 - r_nuc  # (N_e, N_nuc, 3)
        return np.sqrt(np.sum(diff * diff, axis=-1))  # (N_e, N_nuc)

    def compute_term_1e1n(self, term, r1, r_nuc, nuc_groups):
        """Compute electron-nuclear correlation term using vectorized operations."""
        # Check for spin-dependent terms
        if '1=2' not in term['rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
            
        # Get cutoff parameters
        C = term['e-n cutoff']['Constants']['C']
        order = term['e-n basis']['order']
        
        # Get cutoff radii for different nuclear types
        cutoff_L = {}
        for group_key in nuc_groups.keys():
            channel = f'Channel 1-{group_key}'
            if channel in term['e-n cutoff']['Parameters']:
                cutoff_L[group_key] = term['e-n cutoff']['Parameters'][channel]['L'][0]
        
        # Compute all e-n distances at once
        r_en = self._compute_e_n_distances(r1, r_nuc)  # (N_e, N_nuc)
        
        # Initialize result
        total = 0.0
        
        # Process each nuclear group
        for group_key, group_nuclei in nuc_groups.items():
            # Get channel parameters for this group
            channel = f'Channel 1-{group_key}'
            if channel not in term['Linear parameters']:
                continue
                
            channel_params = term['Linear parameters'][channel]
            L = cutoff_L[group_key]
            
            # Get indices for nuclei in this group
            nuc_indices = [int(n[1:])-1 for n in group_nuclei]
            
            # Extract relevant distances
            group_r_en = r_en[:, nuc_indices]  # (N_e, N_group)
            
            # Apply cutoff conditions
            mask = group_r_en < L
            cutoff = np.where(mask, (1 - group_r_en/L)**C, 0.0)
            
            # Process each power term
            for mu in range(2, order + 1):
                param_key = f'c_{mu}'
                if param_key in channel_params:
                    param = channel_params[param_key][0]  # Get parameter value
                    
                    # Compute basis functions and apply cutoff
                    basis = np.where(mask, group_r_en**mu, 0.0)
                    
                    # Sum contributions for this group
                    total += param * np.sum(basis * cutoff)
                    
        return total

    def compute_term_2e1n(self, term, r1, r2, r_nuc, nuc_groups):
        """Compute electron-nuclear-electron correlation terms using vectorized operations."""
        # Check for spin-dependent terms
        if '1=2' not in term['rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
            
        # Get cutoff parameters
        L = term['e-n cutoff']['Constants']['C']
        C = term['e-n cutoff']['Constants']['C']
        order_ee = term['e-e basis']['order']
        order_en = term['e-n basis']['order']
        
        # Compute all required distances at once
        r12 = np.sqrt(np.sum((r1[:, np.newaxis] - r2[np.newaxis])**2, axis=-1))  # (N1, N2)
        
        # Initialize result
        total = 0.0
        
        # Process each nuclear group
        for group_key, group_nuclei in nuc_groups.items():
            # Get channel name for this nuclear group
            channel_name = f'Channel 1-2-{group_key}'
            if channel_name not in term['Linear parameters']:
                continue
                
            channel_params = term['Linear parameters'][channel_name]
            
            # Get L parameter for this nuclear group
            L = term['e-n cutoff']['Parameters'][f'Channel 1-{group_key}']['L'][0]
            
            # Get indices for nuclei in this group
            nuc_indices = [int(n[1:])-1 for n in group_nuclei]
            
            for k in nuc_indices:
                # Compute e-n distances for both electrons
                r1_nuc = np.sqrt(np.sum((r1 - r_nuc[k])**2, axis=-1))  # (N1,)
                r2_nuc = np.sqrt(np.sum((r2 - r_nuc[k])**2, axis=-1))  # (N2,)
                
                # Apply cutoff conditions
                mask1 = r1_nuc < L  # (N1,)
                mask2 = r2_nuc < L  # (N2,)
                cutoff1 = np.where(mask1, (1 - r1_nuc/L)**C, 0.0)  # (N1,)
                cutoff2 = np.where(mask2, (1 - r2_nuc/L)**C, 0.0)  # (N2,)
                
                # For each parameter in the channel
                for param_key, param_info in channel_params.items():
                    if not param_key.startswith('c_'):
                        continue
                        
                    # Extract indices from parameter key (e.g., 'c_1,2,2' -> n=1, l=2, m=2)
                    n, l, m = map(int, param_key[2:].split(','))
                    param_value = param_info[0]  # Get the parameter value
                    
                    # Compute basis functions
                    basis_i = np.where(mask1, r1_nuc**l, 0.0)  # (N1,)
                    basis_j = np.where(mask2, r2_nuc**m, 0.0)  # (N2,)
                    basis_ij = r12**n  # (N1, N2)
                    
                    # Combine all terms
                    contrib = param_value * basis_ij * \
                             (basis_i[:, None] * cutoff1[:, None]) * \
                             (basis_j[None, :] * cutoff2[None, :])
                    
                    total += np.sum(contrib)
        
        return total

    def compute_term_1e2n(self, term, r1, r_nuc, nuc_groups):
        """Compute electron-two-nuclear correlation term using vectorized operations."""
        # Check for spin-dependent terms
        if '1=2' not in term['rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
            
        # Get cutoff parameters
        C = term['e-n cutoff']['Constants']['C']
        
        # Get all cutoff radii for different nuclear types
        cutoff_L = {}
        for group_key in nuc_groups.keys():
            channel = f'Channel 1-{group_key}'
            if channel in term['e-n cutoff']['Parameters']:
                cutoff_L[group_key] = term['e-n cutoff']['Parameters'][channel]['L'][0]
        
        # Compute all e-n distances at once
        r_en = self._compute_e_n_distances(r1, r_nuc)  # (N_e, N_nuc)
        
        # Initialize result
        total = 0.0
        
        # Process each nuclear group pair
        for group1, nuclei1 in nuc_groups.items():
            for group2, nuclei2 in nuc_groups.items():
                if group1 >= group2:  # Skip duplicate pairs and same group
                    continue
                    
                # Get channel parameters for this group pair
                channel = f'Channel 1-{group1}-{group2}'
                if channel not in term['Linear parameters']:
                    continue
                    
                # Get indices for each nuclear group
                indices1 = [int(n[1:])-1 for n in nuclei1]  # Convert 'n1' to 0, etc.
                indices2 = [int(n[1:])-1 for n in nuclei2]  # Convert 'n2' to 1, etc.
                
                L1 = cutoff_L[group1]
                L2 = cutoff_L[group2]
                
                # Extract relevant distances
                r_e1 = r_en[:, indices1]  # (N_e, N_group1)
                r_e2 = r_en[:, indices2]  # (N_e, N_group2)
                
                # Apply cutoff conditions
                mask1 = r_e1 < L1
                mask2 = r_e2 < L2
                cutoff1 = np.where(mask1, (1 - r_e1/L1)**C, 0.0)
                cutoff2 = np.where(mask2, (1 - r_e2/L2)**C, 0.0)
                
                # Loop over all parameter combinations
                channel_params = term['Linear parameters'][channel]
                for param_key, param_info in channel_params.items():
                    if not param_key.startswith('c_'):
                        continue
                    
                    # Extract power indices (e.g., 'c_2,3' -> l=2, m=3)
                    l, m = map(int, param_key[2:].split(','))
                    param_value = param_info[0]  # Get parameter value
                    
                    # Compute basis functions
                    basis1 = np.where(mask1, r_e1**l, 0.0)  # (N_e, N_group1)
                    basis2 = np.where(mask2, r_e2**m, 0.0)  # (N_e, N_group2)
                    
                    # Sum over all nucleus pairs between the groups
                    total += param_value * np.sum(
                        (basis1 * cutoff1)[:, :, None] * 
                        (basis2 * cutoff2)[:, None, :]
                    )
        
        return total