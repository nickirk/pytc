import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from pyscf import gto
from .jastrow import Jastrow

class NuclearCuspJastrow(Jastrow):
    def __init__(self, n_radial=1000):
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
        # Store as list instead of JAX array since shapes may differ
        self.ao_values = ao_values  # Changed from jnp.array(ao_values)
        
        # Instead of MO transformation, just use 1s orbital values
        sao_sums = []
        for atom_id in range(self.n_nuclei):
            s_ao_vals = jnp.array(self.ao_values[atom_id])  # (n_radial, n_s_orbs)
            # For now, just take the first s-orbital (1s) contribution
            # Assuming first s-orbital in the basis set is 1s
            sao_sum = s_ao_vals[:, 0]  # Only use 1s orbital
            sao_sums.append(sao_sum)
        
        # Setup cubic spline interpolators for s-orbital values at each nucleus
        self.splines = []
        for i in range(self.n_nuclei):
            x = np.array(self.r_grids[i])
            y = np.array(sao_sums[i])
            spline = CubicSpline(x, y, bc_type='natural')
            self.splines.append(spline)
            
    def init_params(self):
        """Initialize parameter dictionary structure."""
        # Initialize parameters for each unique nuclear type
        params = {
            'rc': jnp.array([1.0/float(Z) for Z in self.Z_to_idx.keys()]),  # rc = 1/Z for each type
            'poly_coeff': jnp.zeros((self.n_types, 5)),
            'C': jnp.zeros(self.n_types)  # Changed from ones to zeros
        }
        
        # Initialize α coefficients for each nucleus type
        for Z_type, Z_idx in self.Z_to_idx.items():
            Z = float(Z_type)
            rc = params['rc'][Z_idx]
            
            # Find first nucleus of this type
            for i in range(self.n_nuclei):
                if self.charges[i] == Z:
                    nucleus_idx = i
                    break
                    
            phi_rc_vals = self._get_phi_s_derivatives(nucleus_idx, rc)
            phi_0 = self.eval_mo_at_r(nucleus_idx, 1e-8)
            
            # Set up X values
            X = jnp.zeros(5)
            X = X.at[0].set(jnp.log(abs(phi_rc_vals[0])))  # X₁ = ln|φ(rc)|
            X = X.at[1].set(phi_rc_vals[1]/phi_rc_vals[0])  # X₂ = φ'(rc)/φ(rc)
            X = X.at[2].set(phi_rc_vals[2]/phi_rc_vals[0])  # X₃ = φ''(rc)/φ(rc)
            X = X.at[3].set(-Z)  # X₄ = -Z (cusp condition)
            X = X.at[4].set(jnp.log(abs(phi_0)))  # X₅ = ln|φ(0)|
            
            # Compute α coefficients
            alpha = self._compute_alpha_coeffs(Z, rc, X)
            params['poly_coeff'] = params['poly_coeff'].at[Z_idx].set(alpha)
            
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
            (ln(φ_cusp(r)) - ln(φ_s(r)))Θ(r-rc) for each nucleus, summed
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
            contrib = jnp.log(phi_cusp/phi_s) * cutoff
            
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
    
    def _get_phi_s_derivatives(self, nucleus_idx, r):
        """Compute φ_s and its derivatives at given r.
        
        Args:
            nucleus_idx: Index of nucleus
            r: Distance from nucleus
            
        Returns:
            tuple (φ_s, φ_s', φ_s'') at r
        """
        # Use spline object to get derivatives
        r_np = np.array(r)
        phi = self.splines[nucleus_idx](r_np)
        phi_d1 = self.splines[nucleus_idx].derivative(1)(r_np)
        phi_d2 = self.splines[nucleus_idx].derivative(2)(r_np)
        return jnp.array([phi, phi_d1, phi_d2])

    def _compute_alpha_coeffs(self, Z, rc, X_vals):
        """Compute α coefficients from X values and rc.
        
        Args:
            Z: Nuclear charge
            rc: Cutoff radius
            X_vals: Array of X1-X5 values
            
        Returns:
            Array of α coefficients [α₀, α₁, α₂, α₃, α₄]
        """
        X1, X2, X3, X4, X5 = X_vals
        
        alpha = jnp.zeros(5)
        # α₀ = X₅
        alpha = alpha.at[0].set(X5)
        # α₁ = X₄
        alpha = alpha.at[1].set(X4)
        # α₂ = 6X₁/rc² - 3X₂/rc + X₃/2 - 3X₄/rc - 6X₅/rc² - X₂²/2
        alpha = alpha.at[2].set(
            6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
        )
        # α₃ = -8X₁/rc³ + 5X₂/rc² - X₃/rc + 3X₄/rc² + 8X₅/rc³ + X₂²/rc
        alpha = alpha.at[3].set(
            -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8*X5/rc**3 + X2**2/rc
        )
        # α₄ = 3X₁/rc⁴ - 2X₂/rc³ + X₃/(2rc²) - X₄/rc³ - 3X₅/rc⁴ - X₂²/(2rc²)
        alpha = alpha.at[4].set(
            3*X1/rc**4 - 2*X2/rc**3 + X3/(2*rc**2) - X4/rc**3 - 3*X5/rc**4 - X2**2/(2*rc**2)
        )
        return alpha
