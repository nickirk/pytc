"""CASINO-format Jastrow factor implementation."""

import numpy as np
from scipy.interpolate import CubicSpline
from . import Jastrow


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
        if '1=2' not in term['Rules']:
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

    def compute_orbital_cusp_correction(self, r, term):
        """Compute orbital cusp correction term.
        
        Args:
            r: Distance from nucleus (float or array)
            term: Dictionary containing orbital cusp parameters
        
        Returns:
            Cusp correction value at r
        """
        if '1=2' not in term['Rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
        constants = term['e-n cutoff']['Constants']
        ngrid = constants['ngrid']
        L_grid = constants['L_grid']
        rc = ngrid * L_grid
        
        # Get tabulated orbital values (note: 1000 points from 0 to 999)
        phi_grid = []
        for i in range(ngrid + 1):  # +1 because we have phi_0 to phi_999
            key = f'phi_{i}'
            if key in constants['Orbital 1']:
                phi_grid.append(constants['Orbital 1'][key])
            else:
                break
        
        # Convert to numpy array for interpolation
        phi_grid = np.array(phi_grid)
        # Generate grid points (1000 points from 0 to rc)
        r_grid = np.linspace(0, rc, len(phi_grid))
        
        # Verify grid spacing
        dr = r_grid[1] - r_grid[0]
        if not np.isclose(dr, L_grid, rtol=1e-10):
            raise ValueError(f"Grid spacing mismatch: {dr} != {L_grid}")
        
        # Use cubic spline interpolation
        from scipy.interpolate import CubicSpline
        spline = CubicSpline(r_grid, np.log(phi_grid))
        
        # Interpolate ln(phi) at r using spline
        r = np.asarray(r)
        ln_phi = spline(r)
        
        # Compute cusp-corrected orbital
        alpha_0 = term.get('alpha_0', 0.0)  # Default to 0 if not specified
        ln_phi_tilde = alpha_0 * r + constants.get('C', 0.0)
        
        # Apply theta function (1 inside rc, 0 outside)
        mask = r < rc
        
        return (ln_phi_tilde - ln_phi) * mask

    def compute_term_1e1n(self, term, r1, r_nuc, nuc_groups):
        """Compute electron-nuclear correlation term using vectorized operations."""
        if '1=2' not in term['Rules']:
            raise NotImplementedError("Spin-dependent terms not yet implemented")
        # Get nuclear charge from term
        Z = term['e-n cutoff']['Constants'].get('Z', None)
        
        # Parse rules
        rules = term['Rules']
        use_Z = 'Z' in rules
        # Get all exclusions (!N1, !N2, etc)
        excluded_groups = []
        for rule in rules:
            if rule.startswith('!N'):
                excluded_groups.append(rule[2:])  # Remove '!N' prefix
        
        # Get cutoff parameters
        if 'e-n basis' in term and term['e-n basis']['Type'] != 'none':
            order = term['e-n basis']['order']
        else:
            order = 0  # For orbital cusp terms
        
        # Get cutoff parameters
        C = term['e-n cutoff']['Constants']['C']
        
        # Get cutoff radii for different nuclear types
        cutoff_L = {}
        for group_key in nuc_groups.keys():
            if group_key in excluded_groups:
                continue
            channel = f'Channel 1-{group_key}'
            if channel in term['e-n cutoff']['Parameters']:
                cutoff_L[group_key] = term['e-n cutoff']['Parameters'][channel]['L'][0]
        
        # Compute all e-n distances at once
        r_en = self._compute_e_n_distances(r1, r_nuc)  # (N_e, N_nuc)
        
        # Initialize result
        total = 0.0
        
        # Add orbital cusp correction if present
        if term['e-n cutoff']['Type'] == 'orbital cusp':
            total += self.compute_orbital_cusp_correction(r_en, term)
            
        # Process each nuclear group
        for group_key, group_nuclei in nuc_groups.items():
            # Skip excluded groups
            if group_key in excluded_groups:
                continue
                
            # If using Z-based grouping, check Z matches
            if use_Z and Z is not None:
                nuc_index = int(group_nuclei[0][1:]) - 1
                if abs(self.mol.atom_charges()[nuc_index] - Z) > 1e-6:
                    continue
                
            # Rest of the computation remains the same
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
        if '1=2' not in term['Rules']:
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
        if '1=2' not in term['Rules']:
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