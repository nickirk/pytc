import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from pyscf import gto
from .jastrow import Jastrow

class NuclearCuspJastrow(Jastrow):
    def __init__(self, n_radial=100):
        """Initialize nuclear cusp correction.
        
        Args:
            n_radial: Number of radial grid points for orbital evaluation
        """
        super().__init__()
        self.n_radial = n_radial
        
    def setup_for_molecule(self, mol, mo_coeff):
        """Setup orbital evaluators for given molecule.
        This is separate from __init__ to avoid storing large arrays.
        
        Args:
            mol: PySCF Mole object
            mo_coeff: Molecular orbital coefficients
        """
        self.coords = mol.atom_coords()
        self.charges = mol.atom_charges()
        self.n_nuclei = len(self.charges)
        
        # Group nuclei by atomic number
        unique_Z = jnp.unique(self.charges)
        self.Z_to_idx = {int(Z): i for i, Z in enumerate(unique_Z)}
        self.n_types = len(unique_Z)
        
        # Create radial grids for each nucleus
        r_grids = []
        ao_values = []
        self.s_indices_per_atom = []
        
        for atom_id in range(self.n_nuclei):
            r_grid = jnp.linspace(1e-8, 5.0, self.n_radial)
            coords = jnp.zeros((self.n_radial, 3))
            coords = coords.at[:,0].set(r_grid)
            coords = coords + self.coords[atom_id]
            
            # Get all basis function indices for this atom
            shell_ids = []
            for i in range(mol.nbas):
                if mol.bas_atom(i) == atom_id:
                    shell_ids.append(i)
            
            # Find s-type shells and their AO indices
            s_shells = []
            ao_idx_start = 0
            for i in range(mol.nbas):
                if i in shell_ids and mol.bas_angular(i) == 0:
                    s_shells.append(i)
                if i < shell_ids[0]:  # Count AOs before this atom
                    nctr = mol.bas_nctr(i)
                    l = mol.bas_angular(i)
                    ao_idx_start += (2*l + 1) * nctr
            
            # Store s-type AO indices for this atom
            s_ao_indices = []
            for shell_id in s_shells:
                nctr = mol.bas_nctr(shell_id)
                for c in range(nctr):
                    s_ao_indices.append(ao_idx_start)
                    ao_idx_start += 1
            self.s_indices_per_atom.append(jnp.array(s_ao_indices))
            
            # Evaluate s-type AOs using shls_slice
            if s_shells:  # Only evaluate if we have s-type shells
                shls_slice = (min(s_shells), max(s_shells) + 1)
                ao_values_r = mol.eval_gto('GTOval_sph', coords, shls_slice=shls_slice)
                # Ensure we have correct shape (ngrids, nao)
                if ao_values_r.ndim == 1:
                    ao_values_r = ao_values_r.reshape(-1, 1)
            else:
                # Create empty array if no s-type orbitals
                ao_values_r = jnp.zeros((self.n_radial, 0))
                
            r_grids.append(r_grid)
            ao_values.append(ao_values_r)
        
        self.r_grids = jnp.array(r_grids)
        self.ao_values = jnp.array(ao_values)
        
        # Transform AO values to MO values and sum them for each nucleus
        mo_sums = []
        for atom_id in range(self.n_nuclei):
            s_ao_vals = self.ao_values[atom_id]  # (n_radial, n_s_orbs)
            # Only take occupied orbitals
            n_occ = mol.nelec[0]  # number of occupied orbitals (RHF)
            mo_vals = jnp.dot(s_ao_vals, mo_coeff[self.s_indices_per_atom[atom_id], :n_occ])
            mo_sum = jnp.sum(mo_vals, axis=1)  # sum over occupied MOs
            mo_sums.append(mo_sum)
        
        # Setup cubic spline interpolators for summed MO values at each nucleus
        self.splines = []
        for i in range(self.n_nuclei):
            # Convert to numpy for scipy interpolation
            x = np.array(self.r_grids[i])
            y = np.array(mo_sums[i])
            spline = CubicSpline(x, y, bc_type='natural')
            self.splines.append(spline)
            
    def init_params(self):
        """Initialize parameter dictionary structure with validation."""
        params = {
            'rc': jnp.ones(self.n_types),  # cutoff radius per nucleus type
            'poly_coeff': jnp.zeros((self.n_types, 5)),  # coefficients per type
            'C': jnp.ones(self.n_types)  # scaling factor per type
        }
        self._validate_params(params)
        return params
    
    def _validate_params(self, params):
        """Validate parameter shapes."""
        assert params['rc'].shape == (self.n_types,), f"rc shape {params['rc'].shape} != {(self.n_types,)}"
        assert params['poly_coeff'].shape == (self.n_types, 5), \
            f"poly_coeff shape {params['poly_coeff'].shape} != {(self.n_types, 5)}"
        assert params['C'].shape == (self.n_types,), f"C shape {params['C'].shape} != {(self.n_types,)}"
    
    def _cutoff_function(self, r, rc):
        """Smooth cutoff function using inverse polynomial.
        
        Args:
            r: Distance from nucleus
            rc: Cutoff radius
        """
        n = 3
        return 1.0 / (1.0 + (r/rc)**n)
    
    def _eval_poly(self, r, coeffs):
        """Evaluate polynomial Σ(coeffs[l]*r^l)."""
        powers = jnp.arange(len(coeffs))
        return jnp.sum(coeffs * (r**powers))
    
    def _compute(self, r1, r2, params):
        """Compute nuclear cusp Jastrow correction.
        
        Args:
            r1: Position of electron (3,)
            r2: Not used, kept for interface consistency
            params: Dictionary containing cusp parameters
        
        Returns:
            ln(φ_cusp(r)) - ln(φ_s(r))Θ(r-rc) for each nucleus, summed
        """
        total = 0.0
        
        for nucleus_idx in range(self.n_nuclei):
            # Get distance from electron to nucleus
            dr = r1 - self.coords[nucleus_idx]
            r = jnp.sqrt(jnp.sum(dr**2))
            
            # Get nucleus type index for parameter lookup
            Z = self.charges[nucleus_idx]
            Z_idx = self.Z_to_idx[int(Z)]
            
            # Get parameters for this nucleus type
            rc = params['rc'][Z_idx]
            poly_coeffs = params['poly_coeff'][Z_idx]
            C = params['C'][Z_idx]
            
            # Compute φ_cusp = exp(poly(r)) + C
            poly_val = self._eval_poly(r, poly_coeffs)
            phi_cusp = jnp.exp(poly_val) + C
            
            # Get φ_s from spline interpolation
            phi_s = self.eval_mo_at_r(nucleus_idx, r)
            
            # Combine using cutoff
            cutoff = self._cutoff_function(r, rc)
            contrib = jnp.log(phi_cusp) - jnp.log(phi_s) * cutoff
            
            total = total + contrib
            
        return total
    
    def eval_mo_at_r(self, nucleus_idx, r):
        """Evaluate sum of MO values at distance r from nucleus using spline interpolation.
        
        Args:
            nucleus_idx: Index of nucleus
            r: Distance from nucleus
            
        Returns:
            Interpolated sum of MO values
        """
        # Convert input to numpy, evaluate, then convert back to jax array
        r_np = np.array(r)
        result = self.splines[nucleus_idx](r_np)
        return jnp.array(result)
